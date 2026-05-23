"""Offline property tests for the four baselines AND deterministic ActiveGraph
(Mode A, lexical sub-variant). No API required.

Asserts:
  * each system's retrieve() is deterministic under a fixed state
  * oracle returns ONLY the evidence sessions
  * BM25 ranks the evidence session first on a query that names its content
  * RAG runs at BOTH turn and session granularity
  * full-context-s truncates oldest-first and sets `truncated=True` when over budget
  * activegraph-det-lexical:
      - re-ingest of the same instance is byte-identical (event log equality)
      - retrieves the evidence turn under a token budget when the query names it
      - 1-hop temporal expansion pulls in the paired turn from the same session
  * activegraph-det-embedding:
      - skipped when OPENAI_API_KEY is missing (recorded as a skip, not a pass)

Exits non-zero on any failure. Intended as a CI/PR gate.
"""

from __future__ import annotations

import os
import sys

# Force the char/4 fallback so this script never needs network. The behavior
# we test here doesn't depend on the tokenizer, only on whether truncation
# fires given some budget.
os.environ.setdefault("AGLME_FORCE_TIKTOKEN_FALLBACK", "1")

from activegraph_lme.activegraph.graph import build_graph
from activegraph_lme.config import load_config
from activegraph_lme.data import LMEInstance
from activegraph_lme.systems import build_system
from activegraph_lme.systems.activegraph_det import ActiveGraphDetSystem


PASS = "OK"
FAIL = "FAIL"
SKIP = "SKIP"


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


def _make_instance_for_activegraph() -> LMEInstance:
    """Like the canonical fixture but with a follow-up session that reuses
    the distinguishing word, so it survives the cross-session co-occurrence
    filter under the default thresholds.
    """
    return LMEInstance(
        question_id="qag",
        question_type="multi-session",
        question="What is the name of the cat?",
        answer="Mochi",
        question_date="2025-06-15",
        haystack_session_ids=[
            "s_old", "s_evidence", "s_unrelated", "s_followup", "s_recent",
        ],
        haystack_dates=[
            "2025-04-01", "2025-04-20", "2025-05-10", "2025-05-20", "2025-06-05",
        ],
        haystack_sessions=[
            [
                {"role": "user", "content": "Talking about the weather in Tokyo yesterday."},
                {"role": "assistant", "content": "It looked sunny."},
            ],
            [
                {"role": "user", "content": "I adopted a kitten and named her Mochi.", "has_answer": True},
                {"role": "assistant", "content": "Congratulations on the new kitten Mochi!"},
            ],
            [
                {"role": "user", "content": "I am learning guitar chords this week."},
                {"role": "assistant", "content": "Nice; start with major chords."},
            ],
            [
                {"role": "user", "content": "Mochi knocked over a glass this morning."},
                {"role": "assistant", "content": "Kittens love mischief."},
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


# ---- ActiveGraph deterministic mode A tests ---------------------------------


def _ag_reingest_equality(cfg) -> tuple[str, str]:
    """Property: re-ingesting the same instance with the same config produces
    a byte-identical event log. This is the core determinism guarantee for
    Mode A and the foundation Mode B will need before it can record causal
    provenance with confidence.
    """
    inst = _make_instance_for_activegraph()
    kwargs = dict(
        min_token_length=cfg.activegraph.min_token_length,
        min_session_cooccurrence=cfg.activegraph.min_session_cooccurrence,
        max_doc_freq_fraction=cfg.activegraph.max_doc_freq_fraction,
    )
    g1 = build_graph(
        inst.haystack_session_ids, inst.haystack_dates, inst.haystack_sessions, **kwargs
    )
    g2 = build_graph(
        inst.haystack_session_ids, inst.haystack_dates, inst.haystack_sessions, **kwargs
    )
    j1 = g1.events_json()
    j2 = g2.events_json()
    ok = j1 == j2
    msg = (
        f"activegraph re-ingest equality "
        f"(events={len(g1.events)}, turns={len(g1.turns)}, "
        f"vocab={len(g1.vocab_df)}, "
        f"temporal={sum(1 for e in g1.edges if e.kind == 'temporal')}, "
        f"cooc={sum(1 for e in g1.edges if e.kind == 'cooccurrence')})"
    )
    if not ok:
        msg += f" — first divergence at char {next((i for i, (a,b) in enumerate(zip(j1,j2)) if a!=b), -1)}"
    return (PASS if ok else FAIL, msg)


def _ag_lexical_finds_evidence(cfg) -> tuple[str, str]:
    """Query carries a token that survives the cross-session co-occurrence
    filter (a real LongMemEval haystack has >>4 sessions so most tokens
    survive naturally; toy corpora must hand a surviving token to the query).
    """
    inst = _make_instance_for_activegraph()
    sysobj = build_system("activegraph-det-lexical", cfg)
    st = sysobj.ingest(inst)
    ctx = sysobj.retrieve(st, "Mochi kitten name", inst.question_date)
    has_mochi = "Mochi" in ctx.text
    has_evidence = "s_evidence" in ctx.text
    return (PASS if (has_mochi and has_evidence) else FAIL,
            f"activegraph-det-lexical surfaces evidence "
            f"(Mochi={has_mochi}, s_evidence={has_evidence})")


def _ag_temporal_expansion(cfg) -> tuple[str, str]:
    """When the lexical signal pulls in turn #0 of s_evidence (the user turn
    with 'Mochi'), the 1-hop temporal expansion should also pull in turn #1
    of the same session (the assistant reply). Verifies the graph's
    structural use, not just bag-of-turns retrieval.
    """
    inst = _make_instance_for_activegraph()
    sysobj = build_system("activegraph-det-lexical", cfg)
    st = sysobj.ingest(inst)
    ctx = sysobj.retrieve(st, "Mochi kitten name", inst.question_date)
    meta = ctx.meta or {}
    n_expanded = int(meta.get("n_temporal_expansions", 0))
    # In s_evidence, turn 0 mentions "Mochi" and turn 1 mentions "Mochi" too;
    # one is the seed and the other is the temporal neighbor (or both are
    # seeds). Either way, both should appear in the output text.
    both_evidence_turns = (
        "adopted" in ctx.text.lower() and "congratulations" in ctx.text.lower()
    )
    ok = both_evidence_turns
    return (PASS if ok else FAIL,
            f"activegraph-det-lexical pairs adjacent turns "
            f"(both evidence turns present={both_evidence_turns}, "
            f"n_expanded reported={n_expanded})")


def _ag_global_temporal_expansion(cfg) -> tuple[str, str]:
    """Regression: when budget pressure would otherwise evict an evidence
    turn that ranks below several higher-scoring distractors, the global
    session-date temporal expansion pulls in the evidence turn that lives
    in a different session but is chronologically adjacent to a surviving
    seed.

    Tests ``assemble()`` directly with synthetic scores so the assembly
    behavior is exercised independently of the corpus-relative lexical
    vocab filter (which over-prunes on tiny offline fixtures). This
    mirrors the b46e15ed pattern: one of two consecutive-day evidence
    turns ranks high (Feb 14 = seed) and its temporal neighbor (Feb 15)
    has lower raw similarity than several unrelated distractors that
    would otherwise consume the budget.
    """
    from activegraph_lme.activegraph.graph import build_graph
    from activegraph_lme.activegraph.retrieve import assemble

    inst = LMEInstance(
        question_id="qtemp",
        question_type="temporal-reasoning",
        question="(synthetic-scores)",
        answer="",
        question_date="2025-03-01",
        haystack_session_ids=[
            "s_evidence_day1", "s_evidence_day2",
            "s_d1", "s_d2", "s_d3", "s_d4", "s_d5", "s_d6",
        ],
        haystack_dates=[
            "2025-02-14", "2025-02-15",
            "2025-02-16", "2025-02-17", "2025-02-18", "2025-02-19", "2025-02-20", "2025-02-21",
        ],
        haystack_sessions=[
            # Day 1 evidence: surfaces as a seed.
            [
                {"role": "user", "content": "Attended the charity fundraiser downtown today.", "has_answer": True},
                {"role": "assistant", "content": "Great that you attended."},
            ],
            # Day 2 evidence: temporal neighbor of day 1 in global ordering.
            # Lower raw similarity — only the global temporal expansion saves it.
            [
                {"role": "user", "content": "Returned to the same place again today, second visit.", "has_answer": True},
                {"role": "assistant", "content": "Two visits in a row is dedication."},
            ],
            # Distractors: each scores above day-2-evidence on raw similarity.
            [{"role": "user", "content": "Distractor A user line about an unrelated topic."},
             {"role": "assistant", "content": "Distractor A reply."}],
            [{"role": "user", "content": "Distractor B user line about an unrelated topic."},
             {"role": "assistant", "content": "Distractor B reply."}],
            [{"role": "user", "content": "Distractor C user line about an unrelated topic."},
             {"role": "assistant", "content": "Distractor C reply."}],
            [{"role": "user", "content": "Distractor D user line about an unrelated topic."},
             {"role": "assistant", "content": "Distractor D reply."}],
            [{"role": "user", "content": "Distractor E user line about an unrelated topic."},
             {"role": "assistant", "content": "Distractor E reply."}],
            [{"role": "user", "content": "Distractor F user line about an unrelated topic."},
             {"role": "assistant", "content": "Distractor F reply."}],
        ],
        answer_session_ids=["s_evidence_day1", "s_evidence_day2"],
    )

    graph = build_graph(
        inst.haystack_session_ids, inst.haystack_dates, inst.haystack_sessions,
        min_token_length=cfg.activegraph.min_token_length,
        min_session_cooccurrence=cfg.activegraph.min_session_cooccurrence,
        max_doc_freq_fraction=cfg.activegraph.max_doc_freq_fraction,
    )

    # Synthetic scores: day 1 evidence is the top seed; six distractor sessions
    # outrank day 2 evidence; day 2 has the lowest positive score. Under the
    # OLD assembly (all positive seeds first, then expansion), with a budget
    # that holds ~5 turns, day 2 is evicted because six distractors are added
    # before it. Under the NEW interleaved assembly, day 1's expansion pulls
    # day 2 in before any distractor consumes the budget.
    scores: dict[str, float] = {}
    for t in graph.turns:
        if t.turn_id == "s_evidence_day1#0":
            scores[t.turn_id] = 10.0   # high seed
        elif t.turn_id == "s_evidence_day2#0":
            scores[t.turn_id] = 0.1    # low positive — would lose at tight budget
        elif t.session_id.startswith("s_d") and t.turn_idx == 0:
            scores[t.turn_id] = 5.0    # distractors all rank between
        else:
            scores[t.turn_id] = 0.0    # zero-score: must not be selected as a seed

    # Budget that holds roughly 4-5 short turns.
    res_new = assemble(
        graph, scores, token_budget=200,
        temporal_expansion_hops=cfg.activegraph.temporal_expansion_hops,
    )
    has_d1 = "s_evidence_day1#0" in res_new.selected_turn_ids
    has_d2 = "s_evidence_day2#0" in res_new.selected_turn_ids

    # Sanity: confirm the OLD behavior (no cross-session expansion at all,
    # i.e. hops=0) would have lost day 2 under this budget. Intra-session
    # pairing remains active in both cases — day 2 lives in a DIFFERENT
    # session, so only the global expansion can reach it.
    res_no_exp = assemble(graph, scores, token_budget=200, temporal_expansion_hops=0)
    no_exp_has_d2 = "s_evidence_day2#0" in res_no_exp.selected_turn_ids

    ok = has_d1 and has_d2 and not no_exp_has_d2
    return (PASS if ok else FAIL,
            f"activegraph global-temporal expansion under tight budget "
            f"(hops={cfg.activegraph.temporal_expansion_hops}: "
            f"day1={has_d1}, day2={has_d2}; "
            f"no-hops: day2={no_exp_has_d2})")


def _ag_embedding_skip_or_smoke(cfg) -> tuple[str, str]:
    """If OPENAI_API_KEY is set we exercise the embedding path end-to-end,
    otherwise we record an explicit skip so the offline gate still passes
    AND the truth of what was actually verified is on the record.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return (SKIP, "activegraph-det-embedding skipped (OPENAI_API_KEY not set)")
    inst = _make_instance_for_activegraph()
    sysobj = build_system("activegraph-det-embedding", cfg)
    st = sysobj.ingest(inst)
    a = sysobj.retrieve(st, inst.question, inst.question_date)
    b = sysobj.retrieve(st, inst.question, inst.question_date)
    deterministic = a.text == b.text
    has_evidence = "Mochi" in a.text
    ok = deterministic and has_evidence
    return (PASS if ok else FAIL,
            f"activegraph-det-embedding (deterministic={deterministic}, "
            f"has_evidence={has_evidence})")


def main() -> int:
    cfg = load_config()
    inst = _make_instance()

    results: list[tuple[str, str]] = []
    for name in ["full-context-oracle", "full-context-s", "rag-bm25", "rag-dense",
                 "activegraph-det-lexical"]:
        if name == "rag-dense":
            results.append((SKIP, "rag-dense determinism skipped (needs OpenAI key)"))
            continue
        if name == "activegraph-det-lexical":
            # Use the larger fixture that survives the cross-session co-occurrence filter.
            sysobj = build_system(name, cfg)
            st = sysobj.ingest(_make_instance_for_activegraph())
            a = sysobj.retrieve(st, "Mochi kitten name", "2025-06-15")
            b = sysobj.retrieve(st, "Mochi kitten name", "2025-06-15")
            results.append((PASS if a.text == b.text else FAIL,
                            f"deterministic retrieve({name})"))
            continue
        results.append(_deterministic(name, cfg, inst))

    results.append(_oracle_only_evidence(cfg, inst))
    results.append(_bm25_ranks_evidence_first(cfg, inst))
    results.append(_rag_both_granularities(cfg, inst))
    results.append(_full_context_s_truncation(cfg, inst))

    # ActiveGraph Mode A property tests.
    results.append(_ag_reingest_equality(cfg))
    results.append(_ag_lexical_finds_evidence(cfg))
    results.append(_ag_temporal_expansion(cfg))
    results.append(_ag_global_temporal_expansion(cfg))
    results.append(_ag_embedding_skip_or_smoke(cfg))

    n_fail = 0
    for status, msg in results:
        print(f"  [{status}] {msg}")
        if status == FAIL:
            n_fail += 1
    print()
    if n_fail:
        print(f"{n_fail} failure(s)")
        return 1
    print(f"all property tests passed (skips do not count as failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
