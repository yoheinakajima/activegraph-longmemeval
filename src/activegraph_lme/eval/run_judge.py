"""Wraps the FROZEN upstream judge (third_party/longmemeval/src/evaluation/).

We never modify upstream. We shell out and parse stdout.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_EVAL = REPO_ROOT / "third_party/longmemeval/src/evaluation/evaluate_qa.py"
UPSTREAM_PRINT = REPO_ROOT / "third_party/longmemeval/src/evaluation/print_qa_metrics.py"


def _assert_qid_sets_match(hyp_path: Path, ref_path: Path) -> None:
    hyps = [json.loads(line) for line in hyp_path.read_text().splitlines() if line.strip()]
    refs = json.loads(ref_path.read_text())
    hyp_ids = {h["question_id"] for h in hyps}
    ref_ids = {r["question_id"] for r in refs}
    missing = hyp_ids - ref_ids
    if missing:
        raise RuntimeError(
            f"Hypothesis contains question_ids absent from reference {ref_path.name}: "
            f"{sorted(list(missing))[:5]}... ({len(missing)} total). "
            f"Silent eval mismatch would corrupt scores; refusing to proceed."
        )


def run_judge(
    run_dir: str | Path,
    reference_path: str | Path,
    judge_short_name: str = "gpt-4o",
) -> dict[str, Any]:
    """Run the upstream judge over <run_dir>/hypotheses.jsonl against the
    given reference file, write outputs into run_dir, and return parsed
    scores.
    """
    run_dir = Path(run_dir)
    reference_path = Path(reference_path)
    hyp_path = run_dir / "hypotheses.jsonl"
    if not hyp_path.exists():
        raise FileNotFoundError(hyp_path)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    _assert_qid_sets_match(hyp_path, reference_path)

    log_path = run_dir / "eval.log"
    with open(log_path, "w") as logf:
        # 1) judge each hypothesis
        proc = subprocess.run(
            [
                sys.executable,
                str(UPSTREAM_EVAL),
                judge_short_name,
                str(hyp_path),
                str(reference_path),
            ],
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            check=True,
        )
        # 2) summary table
        results_file = hyp_path.with_name(
            hyp_path.name + f".eval-results-{judge_short_name}"
        )
        proc2 = subprocess.run(
            [
                sys.executable,
                str(UPSTREAM_PRINT),
                str(results_file),
                str(reference_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            check=True,
            text=True,
        )
        logf.write("\n=== print_qa_metrics.py ===\n")
        logf.write(proc2.stdout)

    scores = _parse_metrics_output(proc2.stdout)
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    return scores


_RE_PER_TYPE = re.compile(r"^\s*([a-z\-]+):\s+(\S+)\s+\((\d+)\)\s*$")
_RE_TASK_AVG = re.compile(r"^Task-averaged Accuracy:\s+(\S+)\s*$")
_RE_OVERALL = re.compile(r"^Overall Accuracy:\s+(\S+)\s*$")
_RE_ABSTAIN = re.compile(r"^Abstention Accuracy:\s+(\S+)\s+\((\d+)\)\s*$")


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_metrics_output(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"per_type": {}}
    for raw in text.splitlines():
        line = raw.rstrip()
        m = _RE_PER_TYPE.match(line)
        if m:
            out["per_type"][m.group(1)] = {
                "accuracy": _safe_float(m.group(2)),
                "n": int(m.group(3)),
            }
            continue
        m = _RE_TASK_AVG.match(line)
        if m:
            out["task_averaged_accuracy"] = _safe_float(m.group(1))
            continue
        m = _RE_OVERALL.match(line)
        if m:
            out["overall_accuracy"] = _safe_float(m.group(1))
            continue
        m = _RE_ABSTAIN.match(line)
        if m:
            out["abstention_accuracy"] = _safe_float(m.group(1))
            out["abstention_n"] = int(m.group(2))
    return out
