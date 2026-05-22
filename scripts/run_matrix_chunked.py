"""Run the baseline matrix in foreground-friendly chunks using --resume.

Each (system, dataset, granularity) run is executed in repeated passes of
--limit CHUNK_SIZE until all 50 smoke questions are answered, then eval'd.
Finally regenerates paper/results_tables.md.

Usage:
    uv run python scripts/run_matrix_chunked.py [--chunk N]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/run.yaml"
SMOKE_IDS = ROOT / "config/smoke_ids.txt"


def cli(*args: str, config: str | None = None) -> str:
    cmd = [sys.executable, "-m", "activegraph_lme.cli"] + list(args)
    if config:
        pass  # config already in args
    res = subprocess.run(cmd, cwd=str(ROOT), check=True, text=True, capture_output=True)
    return res.stdout.strip().splitlines()[-1]


def count_done(run_dir: Path) -> int:
    hyp = run_dir / "hypotheses.jsonl"
    if not hyp.exists():
        return 0
    with open(hyp) as f:
        return sum(1 for line in f if line.strip())


def with_granularity(granularity: str) -> Path:
    base = yaml.safe_load(CONFIG.read_text())
    base["retrieval"]["granularity"] = granularity
    tmp = ROOT / f"config/_matrix_{granularity}.yaml"
    tmp.write_text(yaml.safe_dump(base, sort_keys=False))
    return tmp


def run_until_done(
    run_id: str,
    system: str,
    dataset: str,
    config_path: str,
    total: int,
    chunk: int,
) -> str:
    """Keep calling the CLI (with --resume after the first pass) until all
    `total` questions are answered. Returns the run_dir path string."""
    tag = "smoke"
    run_dir = ROOT / "runs" / f"{run_id}__{system}__{dataset}__{tag}"

    done = count_done(run_dir)
    first_pass = done == 0

    while done < total:
        remaining = total - done
        batch = min(chunk, remaining)
        print(f"    [{system}/{dataset}] {done}/{total} done — running next {batch}")

        extra_args = []
        if not first_pass:
            extra_args.append("--resume")
        first_pass = False

        cli(
            "run",
            "--system", system,
            "--dataset", dataset,
            "--smoke",
            "--limit", str(done + batch),
            "--run-id", run_id,
            "--config", config_path,
            "--allow-charfallback",
            *extra_args,
        )

        new_done = count_done(run_dir)
        if new_done <= done:
            print(f"    WARNING: no progress (still {done}). Stopping to avoid loop.")
            break
        done = new_done

    print(f"    [{system}/{dataset}] complete: {done}/{total}")
    return str(run_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=20,
                    help="Max questions per CLI call (default 20, ~80s at 4s/q).")
    args = ap.parse_args()

    smoke_ids = [
        line.strip()
        for line in SMOKE_IDS.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    total = len(smoke_ids)

    cfg = yaml.safe_load(CONFIG.read_text())
    systems = [s for s in cfg["systems"] if s != "activegraph"]
    datasets = list(cfg["datasets"].keys())

    matrix_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: list[dict] = []

    for ds in datasets:
        for sys_name in systems:
            granularities = ["session", "turn"] if sys_name.startswith("rag-") else ["n/a"]
            for g in granularities:
                cfg_path = str(with_granularity(g)) if g != "n/a" else str(CONFIG)
                run_id = f"{matrix_id}__{sys_name}__{ds}__g-{g}"
                print(f"\n==> run: {sys_name} on {ds} granularity={g} (smoke/{total} q)")
                run_dir_str = run_until_done(
                    run_id=run_id,
                    system=sys_name,
                    dataset=ds,
                    config_path=cfg_path,
                    total=total,
                    chunk=args.chunk,
                )
                run_dir = Path(run_dir_str)

                print(f"==> eval: {run_dir_str}")
                cli("eval", "--run-dir", run_dir_str, "--config", cfg_path)

                manifest = json.loads((run_dir / "manifest.json").read_text())
                scores = json.loads((run_dir / "scores.json").read_text())
                pq = manifest["queries"]
                n = max(1, len(pq))
                results.append({
                    "system": sys_name,
                    "dataset": ds,
                    "granularity": g,
                    "run_dir": run_dir_str,
                    "overall_accuracy": scores.get("overall_accuracy"),
                    "task_averaged_accuracy": scores.get("task_averaged_accuracy"),
                    "abstention_accuracy": scores.get("abstention_accuracy"),
                    "per_type": scores.get("per_type", {}),
                    "n_questions": manifest["n_questions"],
                    "n_truncated": manifest["n_truncated"],
                    "mean_prompt_tokens": round(sum(q["prompt_tokens"] for q in pq) / n, 1),
                    "mean_completion_tokens": round(sum(q["completion_tokens"] for q in pq) / n, 1),
                    "mean_context_tokens": round(sum(q["context_tokens"] for q in pq) / n, 1),
                    "wall_clock_s": manifest["wall_clock_s"],
                    "reader_model_resolved": manifest["reader_model_resolved"],
                })

    out = ROOT / "runs" / f"matrix_{matrix_id}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nmatrix summary written: {out}")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_results_table.py"), str(out)],
        check=True,
    )
    print("paper/results_tables.md regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
