"""Run the full system × dataset × granularity matrix and regenerate
paper/results_tables.md.

Modes:
  - default: smoke (50 frozen question_ids, config/smoke_ids.txt)
  - --full:  every question in each dataset

Systems and datasets come from config/run.yaml. RAG baselines are run at
both turn and session granularity so reviewers can see the axis upstream
LongMemEval highlights.
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


def aglme(*args: str) -> str:
    # Stream child output to a temp file rather than capture_output=True.
    # A 500-question cell emits enough stdout to fill the OS pipe buffer (~64KB);
    # with capture_output the parent only drains the pipe after the child exits,
    # so the child blocks writing to a full pipe while the parent blocks waiting
    # for the child -> deadlock at 0% CPU. Writing to a file avoids the pipe.
    import tempfile
    with tempfile.TemporaryFile(mode="w+") as tf:
        subprocess.run(
            [sys.executable, "-m", "activegraph_lme.cli", *args],
            cwd=str(ROOT),
            check=True,
            text=True,
            stdout=tf,
            stderr=subprocess.STDOUT,
        )
        tf.seek(0)
        out = tf.read()
    # `run` writes the run_dir path as the LAST stdout line.
    return out.strip().splitlines()[-1]


def with_granularity(granularity: str) -> Path:
    """Create a side-config file with a specific retrieval granularity."""
    base = yaml.safe_load(CONFIG.read_text())
    base["retrieval"]["granularity"] = granularity
    tmp = ROOT / f"config/_matrix_{granularity}.yaml"
    tmp.write_text(yaml.safe_dump(base, sort_keys=False))
    return tmp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Run every question (default: smoke 50).")
    ap.add_argument(
        "--baselines-only",
        action="store_true",
        help="Skip the ActiveGraph variants; run only the four baselines.",
    )
    ap.add_argument(
        "--activegraph-only",
        action="store_true",
        help="Skip the four baselines; run only the ActiveGraph variants.",
    )
    args = ap.parse_args()
    mode_flag = (
        ["--require-authoritative-tokens"]
        if args.full
        else ["--smoke", "--allow-charfallback"]
    )

    cfg = yaml.safe_load(CONFIG.read_text())
    systems = list(cfg["systems"])
    if args.baselines_only:
        systems = [s for s in systems if not s.startswith("activegraph")]
    if args.activegraph_only:
        systems = [s for s in systems if s.startswith("activegraph")]
    datasets = list(cfg["datasets"].keys())  # ['oracle', 's']

    matrix_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: list[dict] = []

    for ds in datasets:
        for sys_name in systems:
            granularities = ["session", "turn"] if sys_name.startswith("rag-") else ["n/a"]
            for g in granularities:
                cfg_path = with_granularity(g) if g != "n/a" else CONFIG
                run_id = f"{matrix_id}__{sys_name}__{ds}__g-{g}"
                print(f"==> run: {sys_name} on {ds} granularity={g} ({'smoke' if not args.full else 'full'})")
                run_dir = aglme(
                    "run",
                    "--system", sys_name,
                    "--dataset", ds,
                    "--config", str(cfg_path),
                    "--run-id", run_id,
                    *mode_flag,
                )
                print(f"   -> {run_dir}")
                print(f"==> eval: {run_dir}")
                aglme("eval", "--run-dir", run_dir, "--config", str(cfg_path))

                manifest = json.loads(Path(run_dir, "manifest.json").read_text())
                scores = json.loads(Path(run_dir, "scores.json").read_text())
                pq = manifest["queries"]
                n = max(1, len(pq))
                mean_prompt = sum(q["prompt_tokens"] for q in pq) / n
                mean_completion = sum(q["completion_tokens"] for q in pq) / n
                mean_context = sum(q["context_tokens"] for q in pq) / n
                results.append(
                    {
                        "system": sys_name,
                        "dataset": ds,
                        "granularity": g,
                        "run_dir": run_dir,
                        "overall_accuracy": scores.get("overall_accuracy"),
                        "task_averaged_accuracy": scores.get("task_averaged_accuracy"),
                        "abstention_accuracy": scores.get("abstention_accuracy"),
                        "per_type": scores.get("per_type", {}),
                        "n_questions": manifest["n_questions"],
                        "n_truncated": manifest["n_truncated"],
                        "mean_prompt_tokens": round(mean_prompt, 1),
                        "mean_completion_tokens": round(mean_completion, 1),
                        "mean_context_tokens": round(mean_context, 1),
                        "wall_clock_s": manifest["wall_clock_s"],
                        "reader_model_resolved": manifest["reader_model_resolved"],
                    }
                )

    out = ROOT / "runs" / f"matrix_{matrix_id}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nmatrix summary: {out}")

    # Rebuild paper/results_tables.md from this matrix.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_results_table.py"), str(out)],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
