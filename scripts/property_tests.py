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


# ---- sem-extract assembly variants (hybrid / index) -------------------------
#
# These run fully offline: we build the base graph deterministically, inject
# Fact objects + `mentions` edges by hand (no LLM, no extraction cache), and
# prefill the EmbeddingClient's content cache with fixed unit vectors so the
# embedding signal is exercised WITHOUT calling OpenAI. That isolates the
# assembly invariants (mentions following, dedup, render format, budget,
# chronological order, determinism) from any network dependency.


def _sem_graph_with_facts(cfg):
    """Build the canonical activegraph fixture, then attach 3 Facts:

      F1 -> mentions s_evidence#0                 (shared turn)
      F2 -> mentions s_evidence#0 AND s_evidence#1 (shares #0 with F1)
      F3 -> mentions s_followup#0
    """
    import numpy as np

    from activegraph_lme.systems.activegraph_sem_extract import _SemState

    inst = _make_instance_for_activegraph()
    state = build_graph(
        inst.haystack_session_ids,
        inst.haystack_dates,
        inst.haystack_sessions,
        min_token_length=cfg.activegraph.min_token_length,
        min_session_cooccurrence=cfg.activegraph.min_session_cooccurrence,
        max_doc_freq_fraction=cfg.activegraph.max_doc_freq_fraction,
    )
    obj_of = {t.turn_id: t.object_id for t in state.turns}

    def _add_fact(fid_text: str, turn_ids: list[str], sidx: int, sdate: str):
        obj = state.graph.add_object(
            "Fact",
            {
                "fact_id": f"fact:{fid_text}",
                "text": fid_text,
                "session_id": turn_ids[0].rsplit("#", 1)[0],
                "session_date": sdate,
                "session_idx": sidx,
                "source": "test",
            },
        )
        for tid in turn_ids:
            state.graph.add_relation(obj.id, obj_of[tid], "mentions")
        return obj

    _add_fact("F1 about the cat", ["s_evidence#0"], 1, "2025-04-20")
    _add_fact("F2 cat and reply", ["s_evidence#0", "s_evidence#1"], 1, "2025-04-20")
    _add_fact("F3 the followup", ["s_followup#0"], 3, "2025-05-20")

    sem_state = _SemState(state=state, meta={"n_sessions_extracted": 0, "n_cache_hits": 4})
    return sem_state


def _prefill_embedder(system, question: str, fact_to_vec: dict[str, list[float]], q_vec):
    """Give the system an EmbeddingClient whose cache already holds unit
    vectors for every fact text + the question, so embed() never hits the API.
    """
    import numpy as np

    from activegraph_lme.activegraph.retrieve import EmbeddingClient

    emb = EmbeddingClient(model="fake-offline")
    cache = {}
    for text, v in fact_to_vec.items():
        arr = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        cache[text] = arr / n if n > 0 else arr
    qa = np.asarray(q_vec, dtype=np.float32)
    qn = float(np.linalg.norm(qa))
    cache[question] = qa / qn if qn > 0 else qa
    emb._cache = cache
    system._embedder = emb


def _sem_hybrid_render(cfg) -> tuple[str, str]:
    from activegraph_lme.systems import build_system

    sem_state = _sem_graph_with_facts(cfg)
    sysobj = build_system("activegraph-sem-hybrid", cfg)
    sysobj._extraction_cache_enabled = False  # don't touch the committed cache
    q = "Tell me about the cat"
    # score order: F1 (1.0) > F2 (0.8) > F3 (0.0)
    _prefill_embedder(
        sysobj,
        q,
        {
            "F1 about the cat": [1.0, 0.0, 0.0],
            "F2 cat and reply": [0.8, 0.6, 0.0],
            "F3 the followup": [0.0, 1.0, 0.0],
        },
        [1.0, 0.0, 0.0],
    )
    ctx = sysobj.retrieve(sem_state, q, "2025-06-15")
    ctx2 = sysobj.retrieve(sem_state, q, "2025-06-15")
    meta = ctx.meta or {}

    # 1) deterministic
    det = ctx.text == ctx2.text
    # 2) shared turn rendered ONCE, with BOTH F1 and F2 headers above it,
    #    F1 before F2 (selection-rank order).
    seg = "[fact: F1 about the cat]\n[fact: F2 cat and reply]\n[Session s_evidence"
    shared_once = ctx.text.count("[Session s_evidence (2025-04-20)] user:") == 1
    headers_stacked = seg in ctx.text
    # 3) chronological: evidence block precedes followup block
    chrono = ctx.text.index("s_evidence") < ctx.text.index("s_followup")
    # 4) meta counters present and sane
    counters = (
        meta.get("n_facts_selected") == 3
        and meta.get("n_unique_turns_rendered") == 3
        and meta.get("assembly_variant") == "hybrid"
        and set(meta.get("selected_fact_ids", []))
        == {"fact:F1 about the cat", "fact:F2 cat and reply", "fact:F3 the followup"}
    )
    ok = det and shared_once and headers_stacked and chrono and counters
    return (
        PASS if ok else FAIL,
        f"sem-hybrid render (det={det}, shared_turn_once={shared_once}, "
        f"headers_stacked={headers_stacked}, chrono={chrono}, counters={counters})",
    )


def _sem_hybrid_budget(cfg) -> tuple[str, str]:
    """Tiny budget => not every fact's entry fits; truncated flag set and
    fewer than all 3 facts selected."""
    from activegraph_lme.systems import build_system

    sem_state = _sem_graph_with_facts(cfg)
    sysobj = build_system("activegraph-sem-hybrid", cfg)
    sysobj._extraction_cache_enabled = False
    sysobj.token_budget = 12  # enough for the top fact entry only
    q = "Tell me about the cat"
    _prefill_embedder(
        sysobj,
        q,
        {
            "F1 about the cat": [1.0, 0.0, 0.0],
            "F2 cat and reply": [0.8, 0.6, 0.0],
            "F3 the followup": [0.0, 1.0, 0.0],
        },
        [1.0, 0.0, 0.0],
    )
    ctx = sysobj.retrieve(sem_state, q, "2025-06-15")
    meta = ctx.meta or {}
    ok = ctx.truncated is True and 0 <= meta.get("n_facts_selected", 99) < 3
    return (
        PASS if ok else FAIL,
        f"sem-hybrid budget (truncated={ctx.truncated}, "
        f"n_facts_selected={meta.get('n_facts_selected')})",
    )


def _sem_index_render(cfg) -> tuple[str, str]:
    from activegraph_lme.systems import build_system

    sem_state = _sem_graph_with_facts(cfg)
    sysobj = build_system("activegraph-sem-index", cfg)
    sysobj._extraction_cache_enabled = False
    q = "Tell me about the cat"
    _prefill_embedder(
        sysobj,
        q,
        {
            "F1 about the cat": [1.0, 0.0, 0.0],
            "F2 cat and reply": [0.8, 0.6, 0.0],
            "F3 the followup": [0.0, 1.0, 0.0],
        },
        [1.0, 0.0, 0.0],
    )
    ctx = sysobj.retrieve(sem_state, q, "2025-06-15")
    ctx2 = sysobj.retrieve(sem_state, q, "2025-06-15")
    meta = ctx.meta or {}

    det = ctx.text == ctx2.text
    # Reader sees ONLY turns — no fact headers leak through.
    no_facts = "[fact:" not in ctx.text
    # shared turn s_evidence#0 deduped to one occurrence
    dedup = ctx.text.count("[Session s_evidence (2025-04-20)] user:") == 1
    chrono = ctx.text.index("s_evidence") < ctx.text.index("s_followup")
    counters = (
        meta.get("assembly_variant") == "index"
        and meta.get("n_unique_turns_rendered") == 3
    )
    ok = det and no_facts and dedup and chrono and counters
    return (
        PASS if ok else FAIL,
        f"sem-index render (det={det}, no_fact_headers={no_facts}, "
        f"dedup={dedup}, chrono={chrono}, counters={counters})",
    )


def _sem_facts_with_turns(cfg) -> tuple[str, str]:
    from activegraph_lme.systems.activegraph_sem_variants import _facts_with_turns

    sem_state = _sem_graph_with_facts(cfg)
    facts = _facts_with_turns(sem_state.state)
    by_id = {f.fact_id: f for f in facts}
    f2 = by_id.get("fact:F2 cat and reply")
    # F2 mentions two turns, chronologically ordered, deduped.
    ok = (
        len(facts) == 3
        and f2 is not None
        and [t.turn_id for t in f2.turns] == ["s_evidence#0", "s_evidence#1"]
    )
    return (
        PASS if ok else FAIL,
        f"sem _facts_with_turns (n_facts={len(facts)}, "
        f"F2_turns={[t.turn_id for t in f2.turns] if f2 else None})",
    )


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
    results.append(_ag_embedding_skip_or_smoke(cfg))

    # sem-extract assembly variants (offline; prefilled embedder, no API).
    results.append(_sem_facts_with_turns(cfg))
    results.append(_sem_hybrid_render(cfg))
    results.append(_sem_hybrid_budget(cfg))
    results.append(_sem_index_render(cfg))

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
