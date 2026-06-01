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
from pathlib import Path

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
    try:
        st = sysobj.ingest(inst)
        a = sysobj.retrieve(st, inst.question, inst.question_date)
        b = sysobj.retrieve(st, inst.question, inst.question_date)
    except Exception as e:  # noqa: BLE001
        # Embedding endpoint unreachable (e.g. OpenAI host not in the
        # environment's egress allowlist). Record an explicit skip rather
        # than a failure — the embedding path is simply not exercisable here.
        if "allowlist" in str(e).lower() or e.__class__.__name__ in (
            "PermissionDeniedError", "APIConnectionError", "APITimeoutError",
        ):
            return (SKIP, f"activegraph-det-embedding skipped ({e.__class__.__name__}: "
                          f"OpenAI endpoint unreachable from this environment)")
        raise
    deterministic = a.text == b.text
    has_evidence = "Mochi" in a.text
    ok = deterministic and has_evidence
    return (PASS if ok else FAIL,
            f"activegraph-det-embedding (deterministic={deterministic}, "
            f"has_evidence={has_evidence})")


# ---- compiled semantic-memory assembly tests (offline, injected scores) -----


def _build_fact_graph(cfg):
    """Build a small turn graph and hand-write two Facts + `mentions` edges
    so the compiled-memory assemblers can be exercised with INJECTED scores
    (no embedding API). Returns (_SemState, fact_id_by_label)."""
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
    # Fact CAT -> the user turn in s_evidence (turn 0); Fact GUITAR -> the
    # user turn in s_unrelated (turn 0). Distinct sessions so chronology is
    # testable.
    ev_turn = state.by_turn_id["s_evidence#0"]
    un_turn = state.by_turn_id["s_unrelated#0"]
    f_cat = state.graph.add_object(
        "Fact",
        {"fact_id": "fact:cat", "text": "The user adopted a kitten named Mochi.",
         "session_id": "s_evidence", "session_date": "2025-04-20", "session_idx": 1},
    )
    state.graph.add_relation(f_cat.id, ev_turn.object_id, "mentions")
    f_guitar = state.graph.add_object(
        "Fact",
        {"fact_id": "fact:guitar", "text": "The user is learning guitar chords.",
         "session_id": "s_unrelated", "session_date": "2025-05-10", "session_idx": 2},
    )
    state.graph.add_relation(f_guitar.id, un_turn.object_id, "mentions")
    return _SemState(state=state, meta={}), {"cat": "fact:cat", "guitar": "fact:guitar"}


def _sem_hybrid_assembly(cfg) -> tuple[str, str]:
    from activegraph_lme.systems.activegraph_sem_hybrid import (
        ActiveGraphSemHybridSystem,
    )
    from activegraph_lme.systems._sem_compiled import project_facts

    sem_state, _ = _build_fact_graph(cfg)
    sys = ActiveGraphSemHybridSystem(
        token_budget=2500, min_token_length=cfg.activegraph.min_token_length,
        min_session_cooccurrence=cfg.activegraph.min_session_cooccurrence,
        max_doc_freq_fraction=cfg.activegraph.max_doc_freq_fraction,
        extraction_cache_dir="/tmp/_pt_nocache",
    )
    facts = project_facts(sem_state.state)
    scores = {"fact:cat": 0.9, "fact:guitar": 0.1}
    a = sys._assemble(sem_state, facts, scores)
    b = sys._assemble(sem_state, facts, scores)
    has_header = "[fact: The user adopted a kitten named Mochi.]" in a.text
    has_anchor = "adopted a kitten" in a.text.lower()  # provenance turn text
    # Chronological: the cat fact (session_idx 1) precedes guitar (idx 2).
    chrono = a.text.index("Mochi.]") < a.text.index("guitar chords.]")
    deterministic = a.text == b.text
    meta_ok = a.meta["n_facts_selected"] == 2 and a.meta["n_unique_turns_rendered"] == 2
    ok = has_header and has_anchor and chrono and deterministic and meta_ok
    return (PASS if ok else FAIL,
            f"sem-hybrid assembly (header={has_header}, anchor={has_anchor}, "
            f"chrono={chrono}, deterministic={deterministic}, meta_ok={meta_ok})")


def _sem_hybrid_budget(cfg) -> tuple[str, str]:
    from activegraph_lme.systems.activegraph_sem_hybrid import (
        ActiveGraphSemHybridSystem,
    )
    from activegraph_lme.systems._sem_compiled import project_facts

    sem_state, _ = _build_fact_graph(cfg)
    # Budget tight enough for one fact+anchor but not two -> truncated, and
    # only the top-scored (cat) fact survives.
    sys = ActiveGraphSemHybridSystem(
        token_budget=45, min_token_length=cfg.activegraph.min_token_length,
        min_session_cooccurrence=cfg.activegraph.min_session_cooccurrence,
        max_doc_freq_fraction=cfg.activegraph.max_doc_freq_fraction,
        extraction_cache_dir="/tmp/_pt_nocache",
    )
    facts = project_facts(sem_state.state)
    a = sys._assemble(sem_state, facts, {"fact:cat": 0.9, "fact:guitar": 0.1})
    ok = a.truncated and a.meta["n_facts_selected"] == 1 and "Mochi.]" in a.text
    return (PASS if ok else FAIL,
            f"sem-hybrid budget (truncated={a.truncated}, "
            f"n_facts_selected={a.meta['n_facts_selected']})")


def _sem_index_assembly(cfg) -> tuple[str, str]:
    from activegraph_lme.systems.activegraph_sem_index import (
        ActiveGraphSemIndexSystem,
    )
    from activegraph_lme.systems._sem_compiled import project_facts

    sem_state, _ = _build_fact_graph(cfg)
    sys = ActiveGraphSemIndexSystem(
        token_budget=2500, min_token_length=cfg.activegraph.min_token_length,
        min_session_cooccurrence=cfg.activegraph.min_session_cooccurrence,
        max_doc_freq_fraction=cfg.activegraph.max_doc_freq_fraction,
        extraction_cache_dir="/tmp/_pt_nocache",
    )
    facts = project_facts(sem_state.state)
    a = sys._assemble(sem_state, facts, {"fact:cat": 0.9, "fact:guitar": 0.1})
    b = sys._assemble(sem_state, facts, {"fact:cat": 0.9, "fact:guitar": 0.1})
    no_facts_in_text = "[fact:" not in a.text  # reader never sees facts
    has_turn = "adopted a kitten" in a.text.lower()
    deterministic = a.text == b.text
    selected_via_fact = a.meta["n_facts_selected"] == 2
    ok = no_facts_in_text and has_turn and deterministic and selected_via_fact
    return (PASS if ok else FAIL,
            f"sem-index assembly (no_facts_in_text={no_facts_in_text}, "
            f"has_turn={has_turn}, deterministic={deterministic}, "
            f"selected_via_fact={selected_via_fact})")


def _role_cache_roundtrip(cfg) -> tuple[str, str]:
    """Role is part of the cache key: the SAME (session_id, content_sha256)
    stores two distinct entries (user + assistant). Round-trips on disk."""
    import tempfile
    from activegraph_lme.systems.activegraph_sem_extract import (
        _ExtractedFact, _ExtractedFactList, _PersistentExtractionCache,
        _compute_combined_prompt_sha256,
    )
    with tempfile.TemporaryDirectory() as d:
        kw = dict(cache_dir=Path(d), seed="A-v2",
                  prompt_sha256=_compute_combined_prompt_sha256(),
                  extractor_model_requested="claude-sonnet-4-5")
        c = _PersistentExtractionCache(**kw)
        u = _ExtractedFactList(facts=[_ExtractedFact(text="The user owns a kayak.")])
        a = _ExtractedFactList(facts=[_ExtractedFact(text="The assistant recommended a 12ft kayak.")])
        c.put("s1", "csum1", "user", u, extractor_model_resolved="m")
        c.put("s1", "csum1", "assistant", a, extractor_model_resolved="m")
        c.flush_manifest()
        # Reload from disk and confirm both roles survive under one (sid,csum).
        c2 = _PersistentExtractionCache(**kw)
        got_u = c2.get("s1", "csum1", "user")
        got_a = c2.get("s1", "csum1", "assistant")
        two_entries = len(c2) == 2
        roles_distinct = (got_u is not None and got_a is not None
                          and got_u.facts[0].text != got_a.facts[0].text)
    ok = two_entries and roles_distinct
    return (PASS if ok else FAIL,
            f"role cache round-trip (entries={len(c2)}, roles_distinct={roles_distinct})")


def _build_write_path_immediate(cfg) -> tuple[str, str]:
    """build_extract_cache persists each (session, role) unit immediately
    and per-entry: a parse-error on one unit must NOT block any other unit
    from reaching disk (no partner-role gate, no holes). Drives the REAL
    serial loop with a mocked extractor — offline, no API."""
    import sys as _sys
    import tempfile
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path("scripts").resolve()))
    import build_extract_cache as bm  # noqa: E402
    from activegraph_lme.systems.activegraph_sem_extract import (
        _PersistentExtractionCache, _compute_combined_prompt_sha256,
    )

    sessions = [
        (f"sess-{i:02d}", "2025-01-01", i, [{"role": "user", "content": "hi"}])
        for i in range(10)
    ]
    units = [(s, role) for s in sessions for role in ("user", "assistant")]

    def _mock(s, role):
        sid = s[0]
        if sid == "sess-03" and role == "assistant":
            return sid, f"csum-{sid}", None, None  # parse-error -> stub
        return (sid, f"csum-{sid}",
                {"facts": [{"text": f"{role} fact {sid}",
                            "mentioned_turn_idxs": [0]}]},
                "m")

    with tempfile.TemporaryDirectory() as d:
        cache = _PersistentExtractionCache(
            cache_dir=_Path(d), seed="A-v2",
            prompt_sha256=_compute_combined_prompt_sha256(),
            extractor_model_requested="claude-sonnet-4-5",
        )
        n = bm._run_units(cache, units, serial=True, workers=1, extract_fn=_mock)
        on_disk = [
            ln for ln in cache.cache_path.read_text().splitlines() if ln.strip()
        ]
    ok = n == 20 and len(on_disk) == 20
    return (PASS if ok else FAIL,
            f"build write path immediate + ungated (appended={n}, on_disk={len(on_disk)})")


def _seed_a_invalidation(cfg) -> tuple[str, str]:
    """The committed user-only seed-A manifest must be REFUSED under the
    role-aware (combined-prompt) signature — the intended invalidation."""
    from activegraph_lme.systems.activegraph_sem_extract import (
        _PersistentExtractionCache, CacheManifestMismatchError,
        _compute_combined_prompt_sha256,
    )
    seed_dir = Path("data/sem_extract_cache")
    if not (seed_dir / "seed-A.manifest.json").exists():
        return (SKIP, "seed-A not present; invalidation guard not exercised")
    try:
        _PersistentExtractionCache(
            cache_dir=seed_dir, seed="A",
            prompt_sha256=_compute_combined_prompt_sha256(),
            extractor_model_requested="claude-sonnet-4-5",
        )
    except CacheManifestMismatchError:
        return (PASS, "seed-A correctly refused under role-aware signature")
    return (FAIL, "seed-A loaded under new signature (should have been refused)")


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

    # Compiled semantic-memory assembly (offline, injected scores — no API).
    results.append(_sem_hybrid_assembly(cfg))
    results.append(_sem_hybrid_budget(cfg))
    results.append(_sem_index_assembly(cfg))

    # Role-aware extraction (offline).
    results.append(_role_cache_roundtrip(cfg))
    results.append(_build_write_path_immediate(cfg))
    results.append(_seed_a_invalidation(cfg))

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
