"""Diagnostic: for one or more question_ids on a LongMemEval dataset, show
which gold evidence turns (turns marked ``has_answer=True``) survive into
the assembled context produced by an ActiveGraph deterministic retrieval.

This is the validation gate for the temporal-expansion fix in
``src/activegraph_lme/activegraph/retrieve.py``. The lexical variant works
fully offline (no API key required); the embedding variant requires
``OPENAI_API_KEY``.

Usage:
    uv run python scripts/diag_assembly.py \
        --dataset data/longmemeval_s_cleaned.json \
        --question-ids b46e15ed gpt4_a1b77f9c eac54add

Optional flags:
    --signal lexical|embedding   (default: lexical)
    --hops N                     (override config.activegraph.temporal_expansion_hops)
    --token-budget N             (override config.activegraph.token_budget)
    --config path                (default: config/run.yaml)

Output per question_id:
    QID <id> [question_type]
      question: ...
      gold evidence turns (N):
        [PRESENT|MISSING] <session_id>#<turn_idx> (<date>) <role>: <content excerpt>
      assembled: n_seeds=<>, n_temporal_expansions=<>, n_selected=<>,
                 token_budget=<>, truncated=<>
      summary: <K>/<N> evidence turns PRESENT

A turn is considered PRESENT iff its rendered text (the same text the
assembler emits) appears as a substring of the assembled context. This is
deterministic and exact — it does not rely on string normalization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable when run directly via ``python scripts/...``.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from activegraph_lme.activegraph.graph import build_graph  # noqa: E402
from activegraph_lme.activegraph.retrieve import (  # noqa: E402
    assemble,
    EmbeddingClient,
    score_embedding,
    score_lexical,
)
from activegraph_lme.config import load_config  # noqa: E402
from activegraph_lme.data import load_dataset  # noqa: E402


def _render_turn(sid: str, date: str, role: str, content: str) -> str:
    # Mirror activegraph.graph._render_turn exactly. The assembler uses the
    # same rendering, so substring presence is the truth signal we want.
    return f"[Session {sid} ({date})] {role}: {content}"


def _diag_one(inst, *, signal: str, hops: int, token_budget: int, cfg) -> dict:
    graph = build_graph(
        inst.haystack_session_ids,
        inst.haystack_dates,
        inst.haystack_sessions,
        min_token_length=cfg.activegraph.min_token_length,
        min_session_cooccurrence=cfg.activegraph.min_session_cooccurrence,
        max_doc_freq_fraction=cfg.activegraph.max_doc_freq_fraction,
    )

    if signal == "lexical":
        scores = score_lexical(graph, inst.question, min_token_length=cfg.activegraph.min_token_length)
    else:
        embedder = EmbeddingClient(model=cfg.embeddings.model)
        scores, _ = score_embedding(graph, inst.question, embedder)

    res = assemble(
        graph, scores,
        token_budget=token_budget,
        temporal_expansion_hops=hops,
    )

    # Walk the haystack to find gold evidence turns (has_answer=True).
    rows: list[dict] = []
    for s_idx, (sid, date, turns) in enumerate(zip(
        inst.haystack_session_ids, inst.haystack_dates, inst.haystack_sessions
    )):
        for t_idx, turn in enumerate(turns):
            if not turn.get("has_answer"):
                continue
            role = str(turn.get("role", "?"))
            content = str(turn.get("content", ""))
            text = _render_turn(sid, date, role, content)
            present = text in res.text
            rows.append({
                "session_id": sid,
                "turn_idx": t_idx,
                "date": date,
                "role": role,
                "content": content,
                "present": present,
            })

    return {
        "question_id": inst.question_id,
        "question_type": inst.question_type,
        "question": inst.question,
        "n_evidence": len(rows),
        "n_present": sum(1 for r in rows if r["present"]),
        "rows": rows,
        "n_seeds": res.n_seeds,
        "n_temporal_expansions": res.n_expanded,
        "n_selected": len(res.selected_turn_ids),
        "truncated": res.truncated,
        "token_budget": token_budget,
        "hops": hops,
        "signal": signal,
    }


def _print_report(report: dict, *, json_out: bool) -> None:
    if json_out:
        # Strip the noisy `content` field for JSON output by default — keep
        # an excerpt so the reader can still tell which turn.
        out = {**report, "rows": [
            {**r, "content_excerpt": (r["content"][:120] + "…") if len(r["content"]) > 120 else r["content"],
             "content": None}
            for r in report["rows"]
        ]}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"QID {report['question_id']} [{report['question_type']}]")
    print(f"  question: {report['question']!r}")
    print(f"  gold evidence turns ({report['n_evidence']}):")
    if report["n_evidence"] == 0:
        print("    (none flagged with has_answer=True in this instance)")
    else:
        for r in report["rows"]:
            tag = "PRESENT" if r["present"] else "MISSING"
            excerpt = r["content"][:120].replace("\n", " ")
            if len(r["content"]) > 120:
                excerpt += "…"
            print(f"    [{tag}] {r['session_id']}#{r['turn_idx']} "
                  f"({r['date']}) {r['role']}: {excerpt}")
    print(f"  assembled: n_seeds={report['n_seeds']}, "
          f"n_temporal_expansions={report['n_temporal_expansions']}, "
          f"n_selected={report['n_selected']}, "
          f"token_budget={report['token_budget']}, "
          f"truncated={report['truncated']}")
    if report["n_evidence"]:
        print(f"  summary: {report['n_present']}/{report['n_evidence']} evidence turns PRESENT")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/longmemeval_s_cleaned.json",
                   help="Path to the LongMemEval dataset JSON.")
    p.add_argument("--question-ids", nargs="+", required=True,
                   help="One or more question_id values to diagnose.")
    p.add_argument("--signal", choices=("lexical", "embedding"), default="lexical")
    p.add_argument("--hops", type=int, default=None,
                   help="Override activegraph.temporal_expansion_hops.")
    p.add_argument("--token-budget", type=int, default=None,
                   help="Override activegraph.token_budget.")
    p.add_argument("--config", default="config/run.yaml")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    hops = args.hops if args.hops is not None else cfg.activegraph.temporal_expansion_hops
    token_budget = args.token_budget if args.token_budget is not None else cfg.activegraph.token_budget

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}", file=sys.stderr)
        return 2

    instances = load_dataset(dataset_path)
    by_id = {inst.question_id: inst for inst in instances}

    missing = [qid for qid in args.question_ids if qid not in by_id]
    if missing:
        print(f"ERROR: question_ids not found in dataset: {missing}", file=sys.stderr)
        return 2

    n_total_evidence = 0
    n_total_present = 0
    for qid in args.question_ids:
        report = _diag_one(
            by_id[qid],
            signal=args.signal,
            hops=hops,
            token_budget=token_budget,
            cfg=cfg,
        )
        n_total_evidence += report["n_evidence"]
        n_total_present += report["n_present"]
        _print_report(report, json_out=args.json)

    if not args.json:
        print(f"OVERALL: {n_total_present}/{n_total_evidence} evidence turns PRESENT "
              f"(signal={args.signal}, hops={hops}, token_budget={token_budget})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
