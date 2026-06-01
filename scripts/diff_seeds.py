"""Offline fact-level variance between two extraction-cache seeds.

Pure read-only analysis — NO API calls, NO cache rebuilds. Loads two
seed-{A,B,C}.jsonl files, compares their parsed _ExtractedFactList
payloads keyed by (session_id, content_sha256), and reports:

  1. Coverage parity (keys in A, in B, in both, only-in-one).
  2. Per-session fact-set comparison (Jaccard similarity, identical-set
     count, distribution).
  3. Aggregate fact-level divergence (totals, shared, only-A, only-B;
     Jaccard at fact level; symmetric difference vs mean total).
  4. Where variance concentrates: parse-error-stub parity, correlation
     between session fact-count and divergence.
  5. The "stable-core" fraction — how much of each session's fact set
     survives across both extractions.
  6. Side-by-side examples of the 2–3 sessions with the largest
     symmetric difference.

Identity rule: facts within a session are matched by their NORMALIZED
text (``text.strip().lower()``). The system's fact_id is
``sha256(session_id|text)[:16]`` over raw text, so raw-text identity
implies fact_id identity; the normalization is to catch trivial
whitespace/case-only differences without overcounting them as variance
(this is a deliberate widening of the match in A's favor, biases the
"shared" count UP — flagged in the output).

Usage:
    uv run python scripts/diff_seeds.py \
        data/sem_extract_cache/seed-A.jsonl \
        data/sem_extract_cache/seed-B.jsonl \
        --out .scratch_build/seed_diff_A_B.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def load_seed(path: Path) -> tuple[dict[tuple[str, str], list[dict]], int]:
    """Returns (entries_by_key, raw_line_count).

    Duplicate (sid, csum) lines from concurrent workers are deduped
    with last-writer-wins, matching the in-memory cache's load
    semantics."""
    entries: dict[tuple[str, str], list[dict]] = {}
    n_raw = 0
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            n_raw += 1
            obj = json.loads(raw)
            key = (str(obj["session_id"]), str(obj["content_sha256"]))
            facts = list(obj.get("parsed", {}).get("facts") or [])
            entries[key] = facts
    return entries, n_raw


def fact_signatures(facts: list[dict]) -> set[str]:
    """Normalized-text set for a session's fact list."""
    return {_norm(f.get("text", "")) for f in facts if (f.get("text") or "").strip()}


def jaccard(a: set, b: set) -> float:
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def hist_buckets(values: list[float], step: float = 0.1) -> list[tuple[str, int]]:
    """Histogram of Jaccard values in [0, 1] in 0.1 buckets."""
    buckets: dict[str, int] = {}
    edges = [round(i * step, 4) for i in range(int(1 / step) + 1)]
    for v in values:
        # Place 1.0 in the top bucket explicitly.
        if v == 1.0:
            label = f"{edges[-2]:.1f}–1.0"
        else:
            for lo in reversed(edges):
                if v >= lo:
                    label = f"{lo:.1f}–{lo + step:.1f}"
                    break
            else:
                label = f"{0.0:.1f}–{step:.1f}"
        buckets[label] = buckets.get(label, 0) + 1
    # Sort by bucket low edge.
    return sorted(buckets.items(), key=lambda kv: float(kv[0].split("–")[0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed_a", type=Path)
    ap.add_argument("seed_b", type=Path)
    ap.add_argument("--out", type=Path, default=Path(".scratch_build/seed_diff.md"))
    ap.add_argument("--examples", type=int, default=3,
                    help="Number of largest-divergence sessions to print side-by-side.")
    args = ap.parse_args()

    A, A_raw = load_seed(args.seed_a)
    B, B_raw = load_seed(args.seed_b)

    lines: list[str] = []
    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out(f"# Seed diff: {args.seed_a.name} vs {args.seed_b.name}")
    out(f"")
    out(f"Identity rule for fact matching: NORMALIZED `text.strip().lower()`.")
    out(f"Fact-id (`sha256(session_id|text)[:16]`) matches under raw-text equality;")
    out(f"the .strip().lower() widening catches whitespace/case-only variants and")
    out(f"biases the shared count UP slightly (intentional — quantifies variance")
    out(f"net of trivial differences).")
    out(f"")

    # --- 1. Coverage parity --------------------------------------------------
    A_keys = set(A.keys())
    B_keys = set(B.keys())
    inter = A_keys & B_keys
    out("## 1. Coverage parity")
    out("")
    out(f"| | seed-A | seed-B |")
    out(f"|---|---|---|")
    out(f"| raw JSONL lines | {A_raw} | {B_raw} |")
    out(f"| unique (session_id, content_sha256) keys | {len(A_keys)} | {len(B_keys)} |")
    out("")
    out(f"- keys in both:    **{len(inter)}**")
    out(f"- keys only in A:  {len(A_keys - B_keys)}")
    out(f"- keys only in B:  {len(B_keys - A_keys)}")
    if A_keys - B_keys:
        out(f"    sample: {sorted(A_keys - B_keys)[:3]}")
    if B_keys - A_keys:
        out(f"    sample: {sorted(B_keys - A_keys)[:3]}")
    out("")

    # --- 2. Per-session fact-set comparison ---------------------------------
    out("## 2. Per-session fact-set comparison (over the shared key set)")
    out("")
    jacs: list[float] = []
    n_identical = 0
    per_session: list[dict[str, Any]] = []
    for k in sorted(inter):
        sa = fact_signatures(A[k])
        sb = fact_signatures(B[k])
        j = jaccard(sa, sb)
        jacs.append(j)
        identical = (sa == sb)
        if identical:
            n_identical += 1
        per_session.append({
            "key": k, "a_set": sa, "b_set": sb, "jaccard": j,
            "a_n": len(sa), "b_n": len(sb), "shared": len(sa & sb),
            "only_a": len(sa - sb), "only_b": len(sb - sa),
        })

    n_shared_keys = len(inter)
    n_differ = n_shared_keys - n_identical
    out(f"- shared sessions: **{n_shared_keys}**")
    out(f"- identical fact sets: **{n_identical}**  ({n_identical/n_shared_keys*100:.1f} %)")
    out(f"- differing fact sets: **{n_differ}**     ({n_differ/n_shared_keys*100:.1f} %)")
    out("")
    out(f"- per-session Jaccard:")
    out(f"    - mean:   {statistics.mean(jacs):.4f}")
    out(f"    - median: {statistics.median(jacs):.4f}")
    out(f"    - min:    {min(jacs):.4f}")
    out(f"    - max:    {max(jacs):.4f}")
    out("")
    out("Histogram (per-session Jaccard, buckets of 0.1):")
    out("")
    out("| bucket | sessions |")
    out("|---|---|")
    for label, cnt in hist_buckets(jacs):
        out(f"| {label} | {cnt} |")
    out("")

    # --- 3. Aggregate fact-level divergence ---------------------------------
    out("## 3. Aggregate fact-level divergence")
    out("")
    total_A = sum(ps["a_n"] for ps in per_session)
    total_B = sum(ps["b_n"] for ps in per_session)
    shared_facts = sum(ps["shared"] for ps in per_session)
    only_A_facts = sum(ps["only_a"] for ps in per_session)
    only_B_facts = sum(ps["only_b"] for ps in per_session)
    union_facts = shared_facts + only_A_facts + only_B_facts
    mean_total = (total_A + total_B) / 2
    fact_jaccard = shared_facts / union_facts if union_facts else 1.0
    symdiff_over_mean = (only_A_facts + only_B_facts) / mean_total if mean_total else 0.0

    out(f"- total facts in seed-A (over shared sessions): **{total_A}**")
    out(f"- total facts in seed-B (over shared sessions): **{total_B}**")
    out(f"- facts shared (same session + same normalized text): **{shared_facts}**")
    out(f"- facts only in A: {only_A_facts}")
    out(f"- facts only in B: {only_B_facts}")
    out(f"- union: {union_facts}")
    out("")
    out(f"- fact-level Jaccard (shared / union):                   **{fact_jaccard:.4f}**")
    out(f"- symmetric-difference / mean-total ('headline' variance): **{symdiff_over_mean:.4f}** ({symdiff_over_mean*100:.2f} %)")
    out("")

    # --- 4. Where variance concentrates ------------------------------------
    out("## 4. Where variance concentrates")
    out("")
    # 4a. Parse-error stubs.
    a_empty = {k for k in inter if not fact_signatures(A[k])}
    b_empty = {k for k in inter if not fact_signatures(B[k])}
    both_empty = a_empty & b_empty
    out(f"### Parse-error stubs ({{\"facts\": []}})")
    out("")
    out(f"- sessions with empty fact set in A: {len(a_empty)}")
    out(f"- sessions with empty fact set in B: {len(b_empty)}")
    out(f"- both empty:                        **{len(both_empty)}**")
    out(f"- only A empty (B has facts):        {len(a_empty - b_empty)}")
    out(f"- only B empty (A has facts):        {len(b_empty - a_empty)}")
    if a_empty - b_empty:
        out(f"    A-empty-only sample: {sorted(a_empty - b_empty)[:5]}")
    if b_empty - a_empty:
        out(f"    B-empty-only sample: {sorted(b_empty - a_empty)[:5]}")
    out("")

    # 4b. Correlate session fact-count with divergence.
    out("### Divergence vs session fact-count")
    out("")
    out("Mean Jaccard binned by max(|A|,|B|) per session — high-fact sessions are noisier:")
    out("")
    bins: dict[str, list[float]] = defaultdict(list)
    for ps in per_session:
        mx = max(ps["a_n"], ps["b_n"])
        if mx == 0:
            b_label = "0 (both empty)"
        elif mx <= 2:
            b_label = "1–2"
        elif mx <= 5:
            b_label = "3–5"
        elif mx <= 10:
            b_label = "6–10"
        elif mx <= 20:
            b_label = "11–20"
        else:
            b_label = "21+"
        bins[b_label].append(ps["jaccard"])
    out("| max(|A|,|B|) | sessions | mean Jaccard | median Jaccard |")
    out("|---|---|---|---|")
    for label in ["0 (both empty)", "1–2", "3–5", "6–10", "11–20", "21+"]:
        if label in bins:
            vs = bins[label]
            out(f"| {label} | {len(vs)} | {statistics.mean(vs):.4f} | {statistics.median(vs):.4f} |")
    out("")

    # --- 5. Stable-core fraction --------------------------------------------
    out("## 5. Stable-core fraction (per-session)")
    out("")
    out("Fraction of each session's facts that survive across both extractions.")
    out("Defined per-session as `|A ∩ B| / |A ∪ B|` (same as Jaccard, but framed as")
    out("'how much of each session's fact set is core/stable').")
    out("")
    nonempty = [ps for ps in per_session if (ps["a_n"] + ps["b_n"]) > 0]
    cores = [ps["jaccard"] for ps in nonempty]
    out(f"- mean stable-core (over {len(nonempty)} sessions with any facts): **{statistics.mean(cores):.4f}**")
    out(f"- median: {statistics.median(cores):.4f}")
    out(f"- fraction of sessions with stable-core >= 0.5: "
        f"{sum(1 for v in cores if v >= 0.5) / len(cores) * 100:.1f} %")
    out(f"- fraction of sessions with stable-core >= 0.8: "
        f"{sum(1 for v in cores if v >= 0.8) / len(cores) * 100:.1f} %")
    out(f"- fraction of sessions with stable-core == 1.0 (identical): "
        f"{sum(1 for v in cores if v >= 0.9999) / len(cores) * 100:.1f} %")
    out("")
    # Alternative framing: stable-core facts / total-facts at the corpus level.
    corpus_core_a = shared_facts / total_A if total_A else 0.0
    corpus_core_b = shared_facts / total_B if total_B else 0.0
    out(f"- corpus-level stable-core (shared_facts / total_A): **{corpus_core_a:.4f}**")
    out(f"- corpus-level stable-core (shared_facts / total_B): **{corpus_core_b:.4f}**")
    out("")

    # --- 6. Largest-divergence examples -------------------------------------
    out("## 6. Side-by-side: largest-divergence sessions")
    out("")
    # Score: symmetric difference count, ignoring trivial empty-vs-empty.
    scored = [
        (ps["only_a"] + ps["only_b"], ps) for ps in per_session
        if (ps["a_n"] + ps["b_n"]) > 0
    ]
    scored.sort(key=lambda t: -t[0])
    for rank, (score, ps) in enumerate(scored[: args.examples], start=1):
        sid, csum = ps["key"]
        out(f"### Example {rank}: session `{sid}` (content_sha256={csum[:16]}…)")
        out("")
        out(f"  symmetric difference: {score} facts  "
            f"(|A|={ps['a_n']} |B|={ps['b_n']} shared={ps['shared']} "
            f"jaccard={ps['jaccard']:.3f})")
        out("")
        # Order facts: shared first, then only-A, then only-B.
        sa = sorted(ps["a_set"])
        sb = sorted(ps["b_set"])
        shared = sorted(ps["a_set"] & ps["b_set"])
        only_a = sorted(ps["a_set"] - ps["b_set"])
        only_b = sorted(ps["b_set"] - ps["a_set"])
        out(f"  **Shared ({len(shared)})**:")
        for t in shared[:12]:
            out(f"  - {t}")
        if len(shared) > 12:
            out(f"  - ... ({len(shared) - 12} more)")
        out("")
        out(f"  **Only in A ({len(only_a)})**:")
        for t in only_a[:12]:
            out(f"  - {t}")
        if len(only_a) > 12:
            out(f"  - ... ({len(only_a) - 12} more)")
        out("")
        out(f"  **Only in B ({len(only_b)})**:")
        for t in only_b[:12]:
            out(f"  - {t}")
        if len(only_b) > 12:
            out(f"  - ... ({len(only_b) - 12} more)")
        out("")

    # Write report.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
