"""Generate paper/results_tables.md from a matrix summary produced by
run_matrix.py.

Accuracy and mean tokens/query are reported together so the cost axis is
never lost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/results_tables.md"

QTYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


def fmt_acc(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "—"


def render(rows: list[dict]) -> str:
    lines: list[str] = ["# Results", ""]
    if not rows:
        lines.append("_no rows_")
        return "\n".join(lines) + "\n"
    resolved = sorted({r["reader_model_resolved"] for r in rows if r.get("reader_model_resolved")})
    lines.append(f"Reader model (resolved): `{', '.join(resolved) or 'unknown'}`")
    lines.append("")

    # Headline table grouped by dataset.
    datasets = sorted({r["dataset"] for r in rows})
    for ds in datasets:
        lines.append(f"## Dataset: `{ds}`")
        lines.append("")
        lines.append(
            "| system | granularity | overall acc | task-avg acc | abstain acc | "
            "mean ctx tok | mean prompt tok | mean compl tok | n truncated | n |"
        )
        lines.append(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for r in [r for r in rows if r["dataset"] == ds]:
            lines.append(
                f"| {r['system']} | {r['granularity']} | "
                f"{fmt_acc(r['overall_accuracy'])} | "
                f"{fmt_acc(r['task_averaged_accuracy'])} | "
                f"{fmt_acc(r['abstention_accuracy'])} | "
                f"{r['mean_context_tokens']:.0f} | "
                f"{r['mean_prompt_tokens']:.0f} | "
                f"{r['mean_completion_tokens']:.0f} | "
                f"{r['n_truncated']} | {r['n_questions']} |"
            )
        lines.append("")

        # Per-type accuracy table.
        lines.append("### Per-type accuracy")
        lines.append("")
        header = "| system | granularity | " + " | ".join(QTYPES) + " |"
        sep = "|---|---|" + "|".join(["---:"] * len(QTYPES)) + "|"
        lines.append(header)
        lines.append(sep)
        for r in [r for r in rows if r["dataset"] == ds]:
            cells = [
                fmt_acc(r["per_type"].get(t, {}).get("accuracy"))
                for t in QTYPES
            ]
            lines.append(
                f"| {r['system']} | {r['granularity']} | " + " | ".join(cells) + " |"
            )
        lines.append("")

    lines.append("---")
    lines.append("Each cell is a single run; the corresponding `runs/<run_id>/manifest.json` "
                 "pins the repo SHA, submodule SHA, dataset SHA-256, exact reader/judge "
                 "snapshots, seed, and per-query token counts.")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: build_results_table.py <matrix.json>", file=sys.stderr)
        return 2
    rows = json.loads(Path(sys.argv[1]).read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(rows))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
