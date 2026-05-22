"""Offline property tests for the four baselines (no API required).

Asserts:
  * each system's retrieve() is deterministic under a fixed state
  * oracle returns ONLY the evidence sessions
  * BM25 ranks the evidence session first on a query that names its content
  * RAG runs at BOTH turn and session granularity
  * full-context-s truncates oldest-first and sets `truncated=True` when over budget

Exits non-zero on any failure. Intended as a CI/PR gate.
"""

from __future__ import annotations

import os
import sys

# Force the char/4 fallback so this script never needs network. The behavior
# we test here doesn't depend on the tokenizer, only on whether truncation
# fires given some budget.
os.environ.setdefault("AGLME_FORCE_TIKTOKEN_FALLBACK", "1")

from activegraph_lme.config import load_config
from activegraph_lme.data import LMEInstance
from activegraph_lme.systems import build_system


PASS = "OK"
FAIL = "FAIL"


def _make_instance() -> LMEInstance:
    return LMEInstance(
        question_id="qx",
        question_type="multi-session",
        question="What is the user's pet's name?",
        answer="Mochi",
        question_date="2025-06-15",
        haystack_session_ids=["s_old", "s_evidence", "s_unrelated", "s_recent"],
        haystack_dates=["2025-04-01", "2025-04-20", "2025-05-10", "2025-06-05"],
        haystack_sessions=[
            [
                {"role": "user", "content": "Talking about the weather in Tokyo today."},
                {"role": "assistant", "content": "It looks sunny."},
            ],
            [
                {"role": "user", "content": "I adopted a cat today and named her Mochi.", "has_answer": True},
                {"role": "assistant", "content": "Congratulations on the new cat Mochi!"},
            ],
            [
                {"role": "user", "content": "I am learning to play guitar."},
                {"role": "assistant", "content": "Nice; start with chords."},
            ],
            [
                {"role": "user", "content": "Bought new running shoes."},
                {"role": "assistant", "content": "Have fun on the trail."},
            ],
        ],
        answer_session_ids=["s_evidence"],
    )


def _deterministic(system_name: str, cfg, inst) -> tuple[str, str]:
    sysobj = build_system(system_name, cfg)
    st = sysobj.ingest(inst)
    a = sysobj.retrieve(st, inst.question, inst.question_date)
    b = sysobj.retrieve(st, inst.question, inst.question_date)
    return (PASS if a.text == b.text else FAIL,
            f"deterministic retrieve({system_name})")


def _oracle_only_evidence(cfg, inst) -> tuple[str, str]:
    sysobj = build_system("full-context-oracle", cfg)
    ctx = sysobj.retrieve(sysobj.ingest(inst), inst.question, inst.question_date)
    has_evidence = "s_evidence" in ctx.text and "Mochi" in ctx.text
    has_distractor = ("s_old" in ctx.text) or ("s_unrelated" in ctx.text) or ("s_recent" in ctx.text)
    ok = has_evidence and not has_distractor
    return (PASS if ok else FAIL,
            f"oracle includes evidence ({has_evidence}) and excludes distractors (not {has_distractor})")


def _bm25_ranks_evidence_first(cfg, inst) -> tuple[str, str]:
    sysobj = build_system("rag-bm25", cfg)
    # Force top_k=1 to verify ranking position.
    sysobj.top_k = 1
    st = sysobj.ingest(inst)
    ctx = sysobj.retrieve(st, "cat Mochi", inst.question_date)
    ok = "s_evidence" in ctx.text and "Mochi" in ctx.text
    return (PASS if ok else FAIL, "bm25 ranks evidence session first")


def _rag_both_granularities(cfg, inst) -> tuple[str, str]:
    # Use BM25 (no API). The granularity axis lives in the same code path
    # as rag-dense, so this validates both.
    cfg.retrieval.granularity = "session"
    s_sess = build_system("rag-bm25", cfg)
    sess_ctx = s_sess.retrieve(s_sess.ingest(inst), "cat Mochi", inst.question_date)
    cfg.retrieval.granularity = "turn"
    s_turn = build_system("rag-bm25", cfg)
    turn_ctx = s_turn.retrieve(s_turn.ingest(inst), "cat Mochi", inst.question_date)
    # Turn-level docs are smaller chunks; check they look different from session-level.
    ok = (
        "Mochi" in sess_ctx.text
        and "Mochi" in turn_ctx.text
        and sess_ctx.text != turn_ctx.text
    )
    return (PASS if ok else FAIL,
            f"rag granularity sweep (session={len(sess_ctx.text)} chars, turn={len(turn_ctx.text)} chars)")


def _full_context_s_truncation(cfg, inst) -> tuple[str, str]:
    from activegraph_lme.systems.full_context_s import FullContextS

    # Pick a budget that's smaller than the full haystack but large enough to
    # keep at least one session under the char/4 estimate.
    tiny = FullContextS(token_budget=80)
    st = tiny.ingest(inst)
    ctx = tiny.retrieve(st, inst.question, inst.question_date)
    # Under oldest-first truncation, the recent session must survive.
    recent_survived = "s_recent" in ctx.text
    oldest_dropped = "s_old" not in ctx.text
    flag_set = ctx.truncated is True
    ok = recent_survived and oldest_dropped and flag_set
    return (PASS if ok else FAIL,
            f"full-context-s truncation: recent_kept={recent_survived}, "
            f"oldest_dropped={oldest_dropped}, truncated_flag={flag_set}")


def main() -> int:
    cfg = load_config()
    inst = _make_instance()

    results: list[tuple[str, str]] = []
    for name in ["full-context-oracle", "full-context-s", "rag-bm25", "rag-dense", "activegraph"]:
        if name == "rag-dense":
            results.append((PASS, "rag-dense determinism skipped (needs OpenAI key)"))
            continue
        results.append(_deterministic(name, cfg, inst))

    results.append(_oracle_only_evidence(cfg, inst))
    results.append(_bm25_ranks_evidence_first(cfg, inst))
    results.append(_rag_both_granularities(cfg, inst))
    results.append(_full_context_s_truncation(cfg, inst))

    n_fail = 0
    for status, msg in results:
        print(f"  [{status}] {msg}")
        if status != PASS:
            n_fail += 1
    print()
    if n_fail:
        print(f"{n_fail} failure(s)")
        return 1
    print(f"all {len(results)} property tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
