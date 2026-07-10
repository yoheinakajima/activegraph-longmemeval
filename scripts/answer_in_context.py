"""Answer-in-context scorer for a completed run.

Joins three sources by question_id:
  * Dataset gold:    haystack turns where ``has_answer`` is True, plus
                     ``answer_session_ids`` as a session-level fallback.
  * Sidecar:         what aic_sidecar.py says retrieval selected
                     (selected_turn_ids and/or selected_session_ids).
  * Per-question correctness: from the upstream judge's
                     ``hypotheses.jsonl.eval-results-<judge_short_name>``
                     (each line carries ``autoeval_label.label``).

Abstention questions (question_id endswith "_abs") are excluded, matching
upstream convention.

Prints a per-type table + overall, writes <run_dir>/aic_results.json.

Usage:
    python scripts/answer_in_context.py <run_dir>
        [--sidecar <path>] [--judge-short-name gpt-4o] [--out <path>]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from activegraph_lme.data import LMEInstance, load_dataset


def _gold_turn_ids(inst: LMEInstance) -> set[str]:
    out: set[str] = set()
    for sid, turns in zip(inst.haystack_session_ids, inst.haystack_sessions):
        for t_idx, turn in enumerate(turns):
            if turn.get("has_answer"):
                out.add(f"{sid}#{t_idx}")
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    # JSON strings may legally contain U+2028/U+2029. ``str.splitlines()``
    # treats those characters as record separators even though JSONL does not.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    out: dict[str, dict[str, Any]] = {}
    for k, rs in by.items():
        n = len(rs)
        hit_turn = sum(1 for r in rs if r["turn_hit"] is True)
        hit_sess = sum(1 for r in rs if r["session_hit"] is True)
        n_turn_evaluable = sum(1 for r in rs if r["turn_hit"] is not None)
        miss_turn = n_turn_evaluable - hit_turn
        miss_sess = n - hit_sess
        # "Reader fumbled even though retrieval surfaced evidence."
        # Prefer turn evidence; if missing, use session evidence.
        reader_failed_with_evidence = sum(
            1
            for r in rs
            if r["correct"] is False
            and (
                (r["turn_hit"] is True)
                or (r["turn_hit"] is None and r["session_hit"] is True)
            )
        )
        out[k] = {
            "n": n,
            "turn_level_evaluable_n": n_turn_evaluable,
            "answer_in_context_turn": round(hit_turn / n_turn_evaluable, 4) if n_turn_evaluable else None,
            "answer_in_context_session": round(hit_sess / n, 4) if n else None,
            "retrieval_miss_turn": miss_turn if n_turn_evaluable else None,
            "retrieval_miss_session": miss_sess,
            "reader_failed_with_evidence": reader_failed_with_evidence,
            "n_correct": sum(1 for r in rs if r["correct"] is True),
            "n_judged": sum(1 for r in rs if r["correct"] is not None),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--sidecar", type=str, default=None)
    ap.add_argument("--judge-short-name", type=str, default=None,
                    help="Defaults to manifest['judge_short_name'].")
    ap.add_argument("--out", type=str, default=None,
                    help="Defaults to <run_dir>/aic_results.json.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    dataset_path = Path(manifest["dataset_path"])
    judge_name = args.judge_short_name or manifest["judge_short_name"]

    sidecar_path = Path(args.sidecar) if args.sidecar else (run_dir / "aic_sidecar.jsonl")
    eval_path = run_dir / f"hypotheses.jsonl.eval-results-{judge_name}"
    if not eval_path.exists():
        raise SystemExit(
            f"Per-question judge output not found at {eval_path}. "
            f"It is produced by `aglme eval` (upstream evaluate_qa.py)."
        )

    instances = {i.question_id: i for i in load_dataset(dataset_path)}
    if sidecar_path.exists():
        sidecar_rows = _read_jsonl(sidecar_path)
    else:
        retrieval_path = run_dir / "retrieval_records.jsonl"
        if not retrieval_path.exists():
            raise SystemExit(
                f"Neither {sidecar_path} nor {retrieval_path} exists. "
                "Run scripts/aic_sidecar.py or enable retrieval artifacts."
            )
        sidecar_rows = []
        for record in _read_jsonl(retrieval_path):
            selected_turn_ids = list((record.get("meta") or {}).get("selected_turn_ids") or [])
            sidecar_rows.append(
                {
                    "question_id": record["question_id"],
                    "selected_turn_ids": selected_turn_ids,
                    "selected_session_ids": sorted(
                        {turn_id.split("#", 1)[0] for turn_id in selected_turn_ids}
                    ),
                }
            )
    sidecar = {r["question_id"]: r for r in sidecar_rows}
    judged = {r["question_id"]: r for r in _read_jsonl(eval_path)}

    rows: list[dict[str, Any]] = []
    diag_printed = False
    for qid in [q["question_id"] for q in manifest["queries"]]:
        inst = instances.get(qid)
        if inst is None or inst.is_abstention:
            continue
        sc = sidecar.get(qid)
        jd = judged.get(qid)
        if sc is None:
            raise SystemExit(f"Sidecar missing question_id {qid}; re-run aic_sidecar.")

        gold_turn = _gold_turn_ids(inst)
        gold_sess = set(inst.answer_session_ids)
        retrieved_turn = set(sc.get("selected_turn_ids") or [])
        retrieved_sess = set(sc.get("selected_session_ids") or [])

        if gold_turn and retrieved_turn:
            turn_hit: bool | None = gold_turn.issubset(retrieved_turn)
        elif gold_turn and not retrieved_turn:
            turn_hit = None
        else:
            turn_hit = None

        session_hit = bool(gold_sess) and gold_sess.issubset(retrieved_sess)

        correct: bool | None = None
        if jd is not None and isinstance(jd.get("autoeval_label"), dict):
            lab = jd["autoeval_label"].get("label")
            if isinstance(lab, bool):
                correct = lab

        if (
            not diag_printed
            and gold_turn
            and retrieved_turn
            and gold_turn.isdisjoint(retrieved_turn)
        ):
            print(
                "[diag] turn-id format check (first non-overlap):\n"
                f"  question_id={qid}\n"
                f"  one gold turn_id    = {sorted(gold_turn)[0]!r}\n"
                f"  one retrieved id    = {sorted(retrieved_turn)[0]!r}",
                file=sys.stderr,
            )
            diag_printed = True

        rows.append(
            {
                "question_id": qid,
                "question_type": inst.question_type,
                "turn_hit": turn_hit,
                "session_hit": session_hit,
                "correct": correct,
                "n_gold_turn": len(gold_turn),
                "n_gold_sess": len(gold_sess),
                "n_retrieved_turn": len(retrieved_turn),
                "n_retrieved_sess": len(retrieved_sess),
            }
        )

    per_type = _aggregate(rows, "question_type")
    overall = _aggregate([{**r, "_overall": "overall"} for r in rows], "_overall")["overall"]

    turn_evaluable_total = sum(1 for r in rows if r["turn_hit"] is not None)
    turn_mode = "turn-level" if turn_evaluable_total > 0 else "session-level (no turn-ID overlap available)"

    print(f"\nRun: {run_dir.name}")
    print(f"System: {manifest['system']}  Granularity: {sidecar[rows[0]['question_id']].get('granularity', 'n/a') if rows else 'n/a'}")
    print(f"Match mode used: {turn_mode}  (n={len(rows)} non-abstention)\n")
    header = f"{'type':<28} {'n':>4} {'aic_turn':>9} {'aic_sess':>9} {'miss_turn':>10} {'miss_sess':>10} {'rfwe':>5} {'acc':>6}"
    print(header)
    print("-" * len(header))
    for k in sorted(per_type):
        v = per_type[k]
        aic_t = "n/a" if v["answer_in_context_turn"] is None else f"{v['answer_in_context_turn']:.4f}"
        aic_s = f"{v['answer_in_context_session']:.4f}" if v["answer_in_context_session"] is not None else "n/a"
        mt = "n/a" if v["retrieval_miss_turn"] is None else f"{v['retrieval_miss_turn']:>10}"
        acc = f"{v['n_correct'] / v['n_judged']:.4f}" if v["n_judged"] else "n/a"
        print(
            f"{k:<28} {v['n']:>4} {aic_t:>9} {aic_s:>9} {mt:>10} {v['retrieval_miss_session']:>10} "
            f"{v['reader_failed_with_evidence']:>5} {acc:>6}"
        )
    print("-" * len(header))
    v = overall
    aic_t = "n/a" if v["answer_in_context_turn"] is None else f"{v['answer_in_context_turn']:.4f}"
    aic_s = f"{v['answer_in_context_session']:.4f}" if v["answer_in_context_session"] is not None else "n/a"
    mt = "n/a" if v["retrieval_miss_turn"] is None else f"{v['retrieval_miss_turn']:>10}"
    acc = f"{v['n_correct'] / v['n_judged']:.4f}" if v["n_judged"] else "n/a"
    print(
        f"{'OVERALL':<28} {v['n']:>4} {aic_t:>9} {aic_s:>9} {mt:>10} {v['retrieval_miss_session']:>10} "
        f"{v['reader_failed_with_evidence']:>5} {acc:>6}"
    )

    out_path = Path(args.out) if args.out else (run_dir / "aic_results.json")
    out_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "system": manifest["system"],
                "granularity": sidecar[rows[0]["question_id"]].get("granularity") if rows else None,
                "judge_short_name": judge_name,
                "n_non_abstention": len(rows),
                "match_mode": turn_mode,
                "per_type": per_type,
                "overall": overall,
                "per_question": rows,
            },
            indent=2,
        )
    )
    print(f"\n[scorer] wrote {out_path}")
    print(
        "\nLegend:\n"
        "  aic_turn   = fraction with gold turn-IDs subset of retrieved turn-IDs\n"
        "  aic_sess   = fraction with gold session-IDs subset of retrieved session-IDs\n"
        "  miss_turn  = #questions where turn-level evidence not retrieved\n"
        "  miss_sess  = #questions where session-level evidence not retrieved\n"
        "  rfwe       = reader_failed_with_evidence (judged wrong even though retrieval hit)\n"
        "  acc        = per-type accuracy from autoeval_label, recomputed here for sanity\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
