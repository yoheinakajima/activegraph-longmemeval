"""Stage-1 overnight pipeline.

Runs the full sequence unattended:
  Step 1: build seed-B and seed-C extraction caches (variance samples;
          NOT committed; gitignored).
  Step 2: fresh matched det-embedding scored smoke run + eval.
  Step 3: scored sem-extract smoke runs on seeds A, B, C + eval each.
  Step 4: AIC sidecar on the matched sem-extract-A vs det-embedding pair.
  Step 5: tabular summary to .scratch_build/overnight_summary.md.

Robustness contract (unattended operation):
  - Log every step with UTC timestamps to .scratch_build/overnight.log.
  - On any step failure, log the error AND CONTINUE to the next
    independent step. Never abort the whole job.
  - Commit + push after each verified milestone so the morning state is
    durable even if the container is reclaimed mid-job. Run-dir
    artifacts (hypotheses + scores + sidecars + manifest) get
    committed. seed-B and seed-C JSONL caches NEVER get committed —
    they stay gitignored.
  - Parse errors during cache build → stub the offending session as
    {"facts": []} (byte-equivalent to the live behavior.failed path).
  - Worker OOM during parallel build → the subsequent sequential
    `build_extract_cache.py` pass picks up the missing shard.
  - All work scoped to the frozen 50-question smoke subset. Never
    touches the full 500.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / ".scratch_build"
LOG_DIR.mkdir(exist_ok=True)
LOG = LOG_DIR / "overnight.log"
SUMMARY = LOG_DIR / "overnight_summary.md"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"[{stamp()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, name: str, *, env_extra: dict | None = None, capture_tail: int = 800) -> subprocess.CompletedProcess | None:
    """Run a subprocess, log its outcome. Never raises — returns None on
    exception. The CALLER decides whether to continue on failure
    (per the no-halt contract)."""
    log(f"--- run: {name}")
    if isinstance(cmd, list):
        log(f"  cmd: {' '.join(cmd)}")
    else:
        log(f"  cmd: {cmd}")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    t0 = time.monotonic()
    try:
        p = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO),
        )
    except Exception as e:  # noqa: BLE001 — robustness over precision
        log(f"  EXCEPTION running {name}: {e!r}")
        return None
    elapsed = time.monotonic() - t0
    with open(LOG, "a") as f:
        if p.stdout:
            f.write(f"  [{name}] stdout tail ({len(p.stdout)} chars):\n")
            f.write(p.stdout[-capture_tail:] + "\n")
        if p.stderr:
            f.write(f"  [{name}] stderr tail ({len(p.stderr)} chars):\n")
            f.write(p.stderr[-capture_tail:] + "\n")
    log(f"  exit={p.returncode}  elapsed={elapsed:.1f}s")
    return p


def smoke_unique_session_keys() -> dict[tuple[str, str], int]:
    """Compute (session_id, full content_sha256) for every unique
    haystack session across the smoke subset. Mirrors the cache key
    the system uses."""
    from activegraph_lme.config import load_config
    from activegraph_lme.data import load_dataset

    cfg = load_config(str(REPO / "config/run.yaml"))
    ids = {
        line.strip()
        for line in (REPO / "config/smoke_ids.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    instances = [i for i in load_dataset(cfg.datasets["s"]) if i.question_id in ids]
    out: dict[tuple[str, str], int] = {}
    for inst in instances:
        for sid, turns in zip(inst.haystack_session_ids, inst.haystack_sessions):
            text = "\n".join(
                f"[turn {ti}] {t.get('role','?')}: {t.get('content','')}"
                for ti, t in enumerate(turns)
            )
            csum = hashlib.sha256(text.encode("utf-8")).hexdigest()
            out[(sid, csum)] = len(turns)
    return out


def cache_unique_keys(jsonl_path: Path) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    if not jsonl_path.exists():
        return seen
    with open(jsonl_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                seen.add((str(obj["session_id"]), str(obj["content_sha256"])))
            except Exception:  # noqa: BLE001
                pass
    return seen


def stub_missing_as_empty(seed: str, missing: set[tuple[str, str]]) -> int:
    """Append {"facts": []} entries for sessions that consistently
    failed extraction (parse_error). Byte-equivalent to the live
    behavior.failed path (no Facts written for that session)."""
    path = REPO / f"data/sem_extract_cache/seed-{seed}.jsonl"
    n = 0
    with open(path, "a") as f:
        for sid, csum in sorted(missing):
            line = json.dumps(
                {"content_sha256": csum, "parsed": {"facts": []}, "session_id": sid},
                sort_keys=True,
            )
            f.write(line + "\n")
            n += 1
            log(f"    STUBBED {sid} csum={csum[:16]} (parse_error consistent)")
    return n


def flush_manifest(seed: str) -> None:
    """Rebuild the manifest + CHECKSUMS sidecar for the given seed."""
    code = (
        "from activegraph_lme.config import load_config; "
        "from activegraph_lme.systems import build_system; "
        f"sys = build_system('activegraph-sem-extract', load_config('config/run.yaml'), extract_seed='{seed}'); "
        "sys._cache.flush_manifest(); "
        "print('flushed n_entries=', len(sys._cache))"
    )
    run(["uv", "run", "python", "-c", code], f"flush_manifest seed={seed}")


def build_and_verify_seed(seed: str, n_workers: int = 8) -> dict[str, Any]:
    """Build the cache for one seed (parallel + sequential recovery +
    stub remaining parse-errors), then verify zero-LLM-call replay.
    Returns a stats dict for the summary."""
    log(f"===== Step 1: building seed-{seed} (NOT committed) =====")
    stats: dict[str, Any] = {"seed": seed, "verify_passed": False}

    # Clean any stale partial artifact for this seed (we're rebuilding
    # from scratch). seed-A files are unchanged.
    if seed != "A":
        for ext in ("jsonl", "manifest.json"):
            p = REPO / f"data/sem_extract_cache/seed-{seed}.{ext}"
            if p.exists():
                p.unlink()
                log(f"  removed stale {p}")

    # 1a) Parallel build.
    run(["bash", "scripts/build_extract_cache_parallel.sh", seed, str(n_workers)],
        f"parallel-build seed-{seed}")

    # 1b) Sequential recovery (catches OOM'd workers + any parse_errors
    # that the parallel pass missed by virtue of a worker dying).
    run(["uv", "run", "python", "scripts/build_extract_cache.py", "--seed", seed],
        f"sequential-recovery seed-{seed}")

    # 1c) Identify still-missing sessions (consistent parse_error). Stub.
    smoke_keys = smoke_unique_session_keys()
    stats["unique_smoke_sessions"] = len(smoke_keys)
    cached = cache_unique_keys(REPO / f"data/sem_extract_cache/seed-{seed}.jsonl")
    missing = set(smoke_keys.keys()) - cached
    stats["missing_after_recovery"] = len(missing)
    if missing:
        log(f"  Step 1c: {len(missing)} sessions still missing — stubbing as {{\"facts\": []}}")
        n = stub_missing_as_empty(seed, missing)
        stats["stubbed"] = n
    else:
        stats["stubbed"] = 0

    # 1d) Flush manifest after any stubbing.
    flush_manifest(seed)

    # 1e) Verify zero-LLM-call replay.
    p = run(
        ["uv", "run", "python", "scripts/verify_extract_cache.py", "--seed", seed],
        f"verify seed-{seed}",
    )
    if p is not None and p.returncode == 0:
        stats["verify_passed"] = True
        log(f"  seed-{seed} verify PASSED ✓")
    else:
        log(f"  seed-{seed} verify FAILED — leaving artifact in place for inspection")

    # Sanity: read final manifest.
    mpath = REPO / f"data/sem_extract_cache/seed-{seed}.manifest.json"
    if mpath.exists():
        with open(mpath) as f:
            m = json.load(f)
        stats["manifest"] = {
            "n_entries": m.get("n_entries"),
            "extractor_model_resolved": m.get("extractor_model_resolved"),
            "cache_file_sha256": m.get("cache_file_sha256"),
        }
    return stats


def git(cmd: list[str], *, quiet_ok: bool = False) -> bool:
    p = run(["git", *cmd], f"git {' '.join(cmd)[:60]}")
    if p is None:
        return False
    if p.returncode != 0:
        if quiet_ok:
            return False
        log(f"  git returned {p.returncode}")
        return False
    return True


def confirm_seed_not_tracked(seed: str) -> None:
    """Belt-and-suspenders: assert seed-B/C files are gitignored."""
    for ext in ("jsonl", "manifest.json"):
        path = f"data/sem_extract_cache/seed-{seed}.{ext}"
        p = run(["git", "check-ignore", "-v", path], f"check-ignore {path}")
        if p is None or p.returncode != 0:
            log(f"  WARN: seed-{seed} file {path} is NOT gitignored")


def scored_run(system: str, seed: str | None, rid: str) -> dict[str, Any]:
    """Run `cli run` + `cli eval` and read scores.json. Returns stats."""
    cmd = [
        "uv", "run", "python", "-m", "activegraph_lme.cli", "run",
        "--system", system,
        "--dataset", "s",
        "--smoke",
        "--run-id", rid,
        "--allow-charfallback",
    ]
    if seed is not None:
        cmd += ["--extract-seed", seed]
    p = run(cmd, f"cli-run {system} seed={seed} rid={rid}")
    run_dir = REPO / "runs" / f"{rid}__{system}__s__smoke"
    stats: dict[str, Any] = {
        "system": system, "seed": seed, "run_id": rid,
        "run_dir": str(run_dir.relative_to(REPO)),
        "run_ok": p is not None and p.returncode == 0,
    }
    if not run_dir.exists():
        log(f"  run_dir {run_dir} does not exist — skipping eval")
        stats["eval_ok"] = False
        return stats

    # Eval — this calls the upstream judge.
    pe = run(
        ["uv", "run", "python", "-m", "activegraph_lme.cli", "eval",
         "--run-dir", str(run_dir)],
        f"cli-eval {run_dir.name}",
    )
    stats["eval_ok"] = pe is not None and pe.returncode == 0
    scores_path = run_dir / "scores.json"
    if scores_path.exists():
        stats["scores"] = json.loads(scores_path.read_text())
    else:
        stats["scores"] = None
    # The "zero extraction LLM calls" property for sem-extract scored
    # runs is pre-verified by scripts/verify_extract_cache.py against
    # each seed; we don't re-check it here because QueryRecord does
    # not surface system meta into manifest.json.
    return stats


def commit_and_push(message: str, paths: list[str]) -> None:
    """Stage explicit paths and create a commit + push. Used between
    milestones so partial work survives a container reclaim."""
    # Safety: never let seed-B/C jsonl/manifest sneak into a commit.
    paths_safe = []
    for p in paths:
        if "seed-B." in p or "seed-C." in p:
            log(f"  REFUSING to stage gitignored variance cache file: {p}")
            continue
        paths_safe.append(p)
    if not paths_safe:
        log(f"  commit_and_push: nothing safe to stage")
        return
    if not git(["add", *paths_safe]):
        log(f"  git add failed for {paths_safe}")
        return
    # If nothing changed, skip commit.
    p = run(["git", "diff", "--cached", "--quiet"], "git diff --cached")
    if p is not None and p.returncode == 0:
        log("  no staged changes; skipping commit")
        return
    if not git(["commit", "-m", message]):
        return
    if not git(["push", "-u", "origin", "sem-extract-stage1"]):
        log("  push failed; retry once after a brief pause")
        time.sleep(5)
        git(["push", "-u", "origin", "sem-extract-stage1"], quiet_ok=True)


def write_summary(state: dict[str, Any]) -> None:
    """Tabulate everything. Do NOT interpret/conclude — just report."""
    lines: list[str] = []
    lines.append(f"# Stage-1 overnight summary")
    lines.append(f"")
    lines.append(f"Generated: {stamp()}")
    lines.append(f"")

    lines.append("## 1. Extraction-cache builds (variance samples)")
    lines.append("")
    lines.append("| seed | unique smoke sessions | missing after recovery | stubbed | verify | resolved snapshot | cache_sha256[:16] |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in ["A", "B", "C"]:
        e = state.get("cache_stats", {}).get(s, {})
        m = e.get("manifest", {}) or {}
        lines.append(
            f"| {s} | {e.get('unique_smoke_sessions','-')} | {e.get('missing_after_recovery','-')} "
            f"| {e.get('stubbed','-')} | {'PASS' if e.get('verify_passed') else 'FAIL/skip'} "
            f"| {m.get('extractor_model_resolved','-')} "
            f"| {str(m.get('cache_file_sha256','-'))[:16]} |"
        )
    lines.append("")
    lines.append("seed-A is committed at 25a093d (pre-existing). seed-B and seed-C are gitignored variance samples.")
    lines.append("")

    lines.append("## 2. Scored runs (smoke)")
    lines.append("")
    lines.append("| run | system | seed | overall_acc | task_avg_acc | abstention_acc | run_ok | eval_ok |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sr in state.get("scored_runs", []):
        sc = sr.get("scores") or {}
        lines.append(
            f"| {sr.get('run_id','-')} | {sr.get('system','-')} | {sr.get('seed','-') or '-'} "
            f"| {sc.get('overall_accuracy','-')} | {sc.get('task_averaged_accuracy','-')} "
            f"| {sc.get('abstention_accuracy','-')} "
            f"| {'YES' if sr.get('run_ok') else 'NO'} "
            f"| {'YES' if sr.get('eval_ok') else 'NO'} |"
        )
    lines.append("")

    lines.append("### Per-category breakdown (overall_accuracy by question_type)")
    lines.append("")
    # Collect categories across runs.
    cats = set()
    for sr in state.get("scored_runs", []):
        sc = sr.get("scores") or {}
        for cat in (sc.get("per_type") or {}).keys():
            cats.add(cat)
    cats_sorted = sorted(cats)
    if cats_sorted:
        header = ["run"] + cats_sorted
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for sr in state.get("scored_runs", []):
            sc = sr.get("scores") or {}
            pt = sc.get("per_type") or {}
            row = [f"{sr.get('system','-')}/{sr.get('seed','-') or 'na'}"]
            for cat in cats_sorted:
                row.append(str((pt.get(cat) or {}).get("accuracy", "-")))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 3. Extraction-variance band (sem-extract overall_accuracy across seeds A/B/C)")
    lines.append("")
    by_seed: dict[str, float | None] = {}
    for sr in state.get("scored_runs", []):
        if sr.get("system") == "activegraph-sem-extract":
            sc = sr.get("scores") or {}
            v = sc.get("overall_accuracy")
            by_seed[sr.get("seed") or "?"] = v
    nums = [v for v in by_seed.values() if isinstance(v, (int, float))]
    if nums:
        spread = max(nums) - min(nums)
        lines.append(f"  seed-A: {by_seed.get('A','-')}")
        lines.append(f"  seed-B: {by_seed.get('B','-')}")
        lines.append(f"  seed-C: {by_seed.get('C','-')}")
        lines.append(f"  spread (max - min): {spread:.4f}")
    else:
        lines.append("  (no sem-extract scored runs completed)")
    lines.append("")

    lines.append("## 4. Matched comparison: sem-extract seed-A vs det-embedding baseline")
    lines.append("")
    sem_a = next((sr for sr in state.get("scored_runs", []) if sr.get("system") == "activegraph-sem-extract" and sr.get("seed") == "A"), None)
    det = next((sr for sr in state.get("scored_runs", []) if sr.get("system") == "activegraph-det-embedding"), None)
    if sem_a and det:
        sa = sem_a.get("scores") or {}
        db = det.get("scores") or {}
        if isinstance(sa.get("overall_accuracy"), (int, float)) and isinstance(db.get("overall_accuracy"), (int, float)):
            delta = sa["overall_accuracy"] - db["overall_accuracy"]
            lines.append(f"  sem-extract seed-A overall_accuracy: {sa['overall_accuracy']}")
            lines.append(f"  det-embedding overall_accuracy:      {db['overall_accuracy']}")
            lines.append(f"  delta (sem-A − det-emb):             {delta:+.4f}")
        else:
            lines.append("  scores incomplete")
    else:
        lines.append("  (one or both matched runs missing)")
    lines.append("")

    lines.append("## 5. AIC (answer-in-context) sidecar")
    lines.append("")
    aic = state.get("aic", {})
    for k, v in aic.items():
        lines.append(f"  - **{k}**: {v}")
    lines.append("")

    lines.append("## 6. Failures / unverified gates")
    lines.append("")
    fails = state.get("failures", [])
    if not fails:
        lines.append("  (none)")
    else:
        for f in fails:
            lines.append(f"  - {f}")
    lines.append("")

    SUMMARY.write_text("\n".join(lines) + "\n")
    log(f"summary written to {SUMMARY}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-builds", action="store_true",
                    help="Skip seed-B/C builds (assume cached).")
    ap.add_argument("--skip-det", action="store_true")
    args = ap.parse_args()

    log("=========== overnight stage-1 START ===========")
    log(f"  cwd: {REPO}")
    log(f"  git head: {subprocess.run(['git','rev-parse','HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()}")

    state: dict[str, Any] = {
        "cache_stats": {},
        "scored_runs": [],
        "aic": {},
        "failures": [],
    }

    # Step 1: seed-B and seed-C cache builds (NOT committed).
    if not args.skip_builds:
        for seed in ("B", "C"):
            try:
                s = build_and_verify_seed(seed, n_workers=8)
                state["cache_stats"][seed] = s
                if not s.get("verify_passed"):
                    state["failures"].append(f"seed-{seed} verify did not pass; "
                                              f"missing={s.get('missing_after_recovery')} "
                                              f"stubbed={s.get('stubbed')}")
            except Exception as e:  # noqa: BLE001
                log(f"  seed-{seed} build EXCEPTION: {e!r}")
                state["failures"].append(f"seed-{seed} build raised {e!r}")
            confirm_seed_not_tracked(seed)
            # Write running summary so morning review has at least partial data.
            try:
                write_summary(state)
            except Exception as e:  # noqa: BLE001
                log(f"  write_summary partial failed: {e!r}")
    else:
        log("--skip-builds: skipping seed-B/C cache builds")

    # Always include seed-A in the cache table for completeness.
    seedA_manifest = REPO / "data/sem_extract_cache/seed-A.manifest.json"
    if seedA_manifest.exists():
        ma = json.loads(seedA_manifest.read_text())
        state["cache_stats"]["A"] = {
            "seed": "A",
            "unique_smoke_sessions": 2345,
            "missing_after_recovery": 0,
            "stubbed": 3,  # historical, committed
            "verify_passed": True,
            "manifest": {
                "n_entries": ma.get("n_entries"),
                "extractor_model_resolved": ma.get("extractor_model_resolved"),
                "cache_file_sha256": ma.get("cache_file_sha256"),
            },
        }

    # Step 2: matched det-embedding scored smoke run.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    det_run = None
    if not args.skip_det:
        try:
            det_run = scored_run("activegraph-det-embedding", None, f"{ts}_overnight_det")
            state["scored_runs"].append(det_run)
            if det_run.get("run_ok") and det_run.get("eval_ok"):
                commit_and_push(
                    "overnight: det-embedding scored smoke run (matched baseline)",
                    [det_run["run_dir"]],
                )
            else:
                state["failures"].append(f"det-embedding run/eval did not both succeed; not committed")
        except Exception as e:  # noqa: BLE001
            log(f"  det-embedding scored run EXCEPTION: {e!r}")
            state["failures"].append(f"det-embedding scored run raised {e!r}")
        try: write_summary(state)
        except Exception: pass
    else:
        log("--skip-det: skipping det-embedding scored run")

    # Step 3: scored sem-extract runs on A/B/C.
    sem_runs = {}
    for seed in ("A", "B", "C"):
        try:
            ts2 = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            sr = scored_run("activegraph-sem-extract", seed, f"{ts2}_overnight_semExt{seed}")
            state["scored_runs"].append(sr)
            sem_runs[seed] = sr
            if sr.get("run_ok") and sr.get("eval_ok"):
                # Only commit; the seed-B/C scored run dirs ARE allowed to be
                # committed (they are run outputs, NOT the gitignored caches).
                commit_and_push(
                    f"overnight: sem-extract scored smoke run (seed-{seed})",
                    [sr["run_dir"]],
                )
            else:
                state["failures"].append(f"sem-extract seed-{seed} run/eval did not both succeed; not committed")
        except Exception as e:  # noqa: BLE001
            log(f"  sem-extract seed-{seed} scored run EXCEPTION: {e!r}")
            state["failures"].append(f"sem-extract seed-{seed} scored run raised {e!r}")
        try: write_summary(state)
        except Exception: pass

    # Step 4: AIC sidecar on the matched pair (sem-extract-A and det-embedding).
    if det_run and sem_runs.get("A") and det_run.get("run_ok") and sem_runs["A"].get("run_ok"):
        for label, sr in (("det-embedding", det_run), ("sem-extract-A", sem_runs["A"])):
            try:
                p = run(
                    ["uv", "run", "python", "scripts/aic_sidecar.py", sr["run_dir"]],
                    f"aic_sidecar {label}",
                )
                ok = p is not None and p.returncode == 0
                state["aic"][label] = "ok" if ok else "failed"
                sidecar = REPO / sr["run_dir"] / "aic_sidecar.jsonl"
                if sidecar.exists():
                    state["aic"][f"{label}_sidecar_lines"] = sum(1 for _ in open(sidecar))
                    commit_and_push(
                        f"overnight: aic_sidecar.jsonl for {label}",
                        [sr["run_dir"]],
                    )
            except Exception as e:  # noqa: BLE001
                log(f"  aic_sidecar {label} EXCEPTION: {e!r}")
                state["failures"].append(f"aic_sidecar {label} raised {e!r}")
    else:
        log("Step 4 skipped: matched pair runs not both successful")
        state["aic"]["status"] = "skipped (matched pair runs missing)"

    # Step 5: final summary write.
    try:
        write_summary(state)
    except Exception as e:  # noqa: BLE001
        log(f"final write_summary failed: {e!r}")

    log("=========== overnight stage-1 DONE ===========")
    log(f"summary at {SUMMARY}")
    log(f"log at {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
