"""Stage-1 semantic-memory ActiveGraph assembly variants.

Two NEW systems, both built directly on top of
:class:`ActiveGraphSemExtractSystem` (same ingest, same committed
seed-A extraction cache, same ``mentions`` Fact->Turn provenance edges).
They differ ONLY in how ``retrieve()`` assembles the reader context:

  * ``activegraph-sem-hybrid`` (variant a) — facts-as-headers +
    provenance-anchored turns. Facts are scored, greedily selected
    under the token budget, and each selected fact pulls in its
    ``mentions`` source turn(s). The reader sees a labeled fact-header
    ABOVE its anchored turn(s).

  * ``activegraph-sem-index`` (variant b) — facts as retrieval signal
    ONLY. Facts are scored and selected, then REPLACED by their
    ``mentions`` source turn(s); the reader never sees a fact. Output
    format is byte-for-byte the same shape as det-embedding's (turns
    joined by ``"\\n\\n"``), so the only thing that differs from
    det-embedding is the retrieval index (facts vs. turn embeddings).

Why these two together
    They bracket the Borges-style "compile, don't replace" argument.
    sem-index isolates "are facts a better retrieval SIGNAL than direct
    turn embedding?" (reader input identical to det-embedding); sem-hybrid
    asks the follow-on "does ALSO putting the fact in context help the
    reader?". The Stage-1 sem-extract loss (72% vs det-embedding 88%)
    concentrated on single-session-user / multi-session, where extraction
    discarded the raw user turns — both variants retain the raw turns via
    the mentions edges, which is the whole point.

Retrieval signal
    Both variants score the Fact pool with the SAME embedding signal the
    det-embedding baseline uses for turns (pinned ``text-embedding-3-small``
    cosine, L2-normalized dot product). This keeps the comparison about
    the retrieval INDEX (facts vs turns), not the signal type. Facts are
    embedded from their stored ``text``; the question is embedded once.
    Embeddings are content-addressed and cached on the shared
    EmbeddingClient so ``retrieve()`` is deterministic across the harness's
    repeat-call check.

Zero extraction calls
    ingest() is inherited unchanged from ActiveGraphSemExtractSystem, so
    with ``--extract-seed A`` every haystack session is a cache hit and
    no extraction LLM call is made. The inherited per-ingest meta
    (``n_sessions_extracted``, ``n_cache_hits``, ``extraction_cache_*``)
    proves it; we merge those into each retrieve()'s meta and add the
    new assembly counters.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..activegraph.graph import IngestState, Turn
from ..activegraph.retrieve import _FACT_SEQ_OFFSET, EmbeddingClient
from ..tokens import count_tokens as _tok_count
from .activegraph_sem_extract import ActiveGraphSemExtractSystem, _SemState
from .base import AssembledContext


log = logging.getLogger(__name__)


# ---- fact + provenance projection -------------------------------------------


@dataclass(frozen=True)
class _FactWithTurns:
    """A Fact plus its ``mentions`` source turns, resolved through the
    package graph.

    ``turns`` is deduplicated by ``turn_id`` and sorted chronologically
    (by Turn.sort_key) so render order is stable. ``sort_key`` mirrors
    the _FactUnit convention in activegraph/retrieve.py: facts sort
    immediately after their source session's turns in a chronological
    pass (str/int/int tuple, totally orderable against Turn.sort_key).
    """

    fact_id: str
    text: str
    sort_key: tuple
    turns: tuple[Turn, ...]


def _facts_with_turns(state: IngestState) -> list[_FactWithTurns]:
    """Project every Fact object into a _FactWithTurns, following the
    ``mentions`` edges (source = Fact object id, target = Turn object id,
    exactly as written by activegraph_sem_extract._write_facts_to_graph).

    Returns ``[]`` when no Fact objects exist (keeps the variants safe on
    a fact-less graph). Turn views are resolved via ``state.by_object_id``;
    mention targets that don't resolve to a known Turn are skipped.
    """
    mentions_by_fact: dict[str, list[str]] = defaultdict(list)
    for r in state.graph.relations(type="mentions"):
        mentions_by_fact[r.source].append(r.target)

    out: list[_FactWithTurns] = []
    for obj in state.graph.objects(type="Fact"):
        data = obj.data or {}
        text = str(data.get("text", ""))
        fact_id = str(data.get("fact_id") or obj.id)
        session_date = str(data.get("session_date", ""))
        session_idx = int(data.get("session_idx", 0))
        try:
            seq = int(obj.id.rsplit("#", 1)[1])
        except (IndexError, ValueError):
            seq = 0
        sort_key = (session_date, session_idx, _FACT_SEQ_OFFSET + seq)

        seen: set[str] = set()
        turns: list[Turn] = []
        for tobj_id in mentions_by_fact.get(obj.id, ()):
            tv = state.by_object_id.get(tobj_id)
            if tv is None or tv.turn_id in seen:
                continue
            seen.add(tv.turn_id)
            turns.append(tv)
        turns.sort(key=lambda t: t.sort_key)
        out.append(
            _FactWithTurns(
                fact_id=fact_id, text=text, sort_key=sort_key, turns=tuple(turns)
            )
        )
    return out


def score_facts_embedding(
    facts: list[_FactWithTurns], question: str, embedder: EmbeddingClient
) -> dict[str, float]:
    """Cosine similarity between the question and each fact's text.

    Mirrors activegraph/retrieve.py::score_embedding for turns: vectors
    are L2-normalized by the EmbeddingClient, so the dot product is the
    cosine. Returns ``{fact_id: sim}``; empty when there are no facts.
    """
    if not facts:
        return {}
    fact_vecs = embedder.embed([f.text for f in facts])
    q_vec = embedder.embed([question])[0]
    raw = fact_vecs @ q_vec
    return {facts[i].fact_id: float(raw[i]) for i in range(len(facts))}


def _rank_facts(
    facts: list[_FactWithTurns], scores: dict[str, float]
) -> list[_FactWithTurns]:
    """(score desc, sort_key asc) — same tie-break order assemble() uses."""
    return sorted(
        facts, key=lambda f: (-scores.get(f.fact_id, 0.0), f.sort_key)
    )


# ---- shared embedder mixin ---------------------------------------------------


class _FactEmbeddingMixin:
    """Lazily-constructed shared EmbeddingClient over the pinned model.

    The client caches by text content, so embedding the same fact text
    across questions (and the repeat-call determinism check) is free and
    byte-identical.
    """

    embedding_model: str
    _embedder: EmbeddingClient | None

    def _get_embedder(self) -> EmbeddingClient:
        if getattr(self, "_embedder", None) is None:
            self._embedder = EmbeddingClient(model=self.embedding_model)
        return self._embedder

    def _score(
        self, state: _SemState, question: str, facts: list[_FactWithTurns]
    ) -> dict[str, float]:
        return score_facts_embedding(facts, question, self._get_embedder())


# ---- variant a: facts + anchored turns --------------------------------------


class ActiveGraphSemHybridSystem(_FactEmbeddingMixin, ActiveGraphSemExtractSystem):
    """``activegraph-sem-hybrid`` — facts-as-headers + anchored turns.

    Budget split (DESIGN CHOICE, documented here as required):
        Unified pool, scored greedy. Each Fact ENTRY costs
        ``len(fact.text) + sum(len(turn.text) for turn in mentions)``
        tokens against the 2,500 budget. High-scored facts therefore
        consume more budget because they bring their turn(s) with them —
        biased toward fewer-but-deeper context, which is the right
        direction for the Borges-style fidelity argument. Shared turns
        are charged once per fact that brings them (the dedup happens at
        render time only), exactly as specified; the rendered context is
        therefore never larger than the budget we charged.

        Cheaper fallback: if ``hybrid_anchor_top_k`` is set (e.g. 10),
        only the top-K SELECTED facts expand their turns; lower-ranked
        selected facts go in headers-only (cost = ``len(fact.text)``).
        Default is ``None`` = "all selected facts bring turns". The
        decision rule from the experiment brief: run the smoke set with
        the default; if mean ``n_facts_selected`` < 5, downgrade to
        top-10-only and re-run. We surface ``n_facts_selected`` and the
        active mode in meta so that decision is data-driven.

    Render format (DESIGN CHOICE):
        Each selected entry renders as ``[fact: <text>]\\n<turn text>`` —
        the fact as a labeled header above its anchored turn(s). When
        multiple facts share a turn, the turn is rendered ONCE with all
        relevant fact-headers stacked above it (headers ordered by
        selection rank). A selected fact with no mentions (or a
        headers-only fact in top-K mode) renders as a standalone
        ``[fact: <text>]`` header. Entries are emitted in chronological
        order by turn ``(session_date, session_idx, turn_idx)`` (orphan
        fact-headers by the fact's own sort_key). Blocks are joined by
        ``"\\n\\n"`` (the same inter-unit separator det-embedding uses).
    """

    name = "activegraph-sem-hybrid"

    def __init__(
        self,
        *,
        embedding_model: str = "text-embedding-3-small",
        hybrid_anchor_top_k: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        # None => every selected fact expands its turns (default).
        # int  => only the first K SELECTED facts expand; rest header-only.
        self.hybrid_anchor_top_k = hybrid_anchor_top_k
        self._embedder: EmbeddingClient | None = None

    def retrieve(
        self, state: _SemState, question: str, question_date: str
    ) -> AssembledContext:
        facts = _facts_with_turns(state.state)
        scores = self._score(state, question, facts)
        ranked = _rank_facts(facts, scores)
        rank_index = {f.fact_id: i for i, f in enumerate(ranked)}

        budget = self.token_budget
        running = 0
        truncated = False
        # selected entries, in selection order: (fact, brings_turns)
        selected: list[tuple[_FactWithTurns, bool]] = []
        n_selected = 0

        for f in ranked:
            # top-K mode: only the first K selected facts expand turns.
            brings_turns = (
                self.hybrid_anchor_top_k is None
                or n_selected < self.hybrid_anchor_top_k
            )
            turn_cost = (
                sum(_tok_count(t.text) for t in f.turns) if brings_turns else 0
            )
            cost = _tok_count(f.text) + turn_cost
            if running + cost > budget:
                truncated = True
                # Mirror assemble(): keep trying smaller/cheaper entries.
                continue
            running += cost
            selected.append((f, brings_turns))
            n_selected += 1

        text, rendered_turn_ids, n_turns_anchored = _render_hybrid(
            selected, rank_index
        )

        meta = {
            **state.meta,
            "assembly_variant": "hybrid",
            "retrieval_signal": "embedding",
            "embedding_model": self.embedding_model,
            "token_budget": self.token_budget,
            "hybrid_anchor_top_k": self.hybrid_anchor_top_k,
            "n_facts_selected": n_selected,
            "n_turns_anchored": n_turns_anchored,
            "n_unique_turns_rendered": len(rendered_turn_ids),
            # Consumed by scripts/aic_sidecar.py (turns that reached the reader).
            "selected_turn_ids": rendered_turn_ids,
            "selected_fact_ids": [f.fact_id for f, _ in selected],
        }
        return AssembledContext(text=text, truncated=truncated, meta=meta)


def _render_hybrid(
    selected: list[tuple[_FactWithTurns, bool]],
    rank_index: dict[str, int],
) -> tuple[str, list[str], int]:
    """Build the hybrid render: each turn once, fact-headers stacked above.

    Returns ``(text, rendered_turn_ids_chrono, n_turns_anchored)`` where
    ``n_turns_anchored`` counts (fact -> turn) anchor links across selected
    facts that brought turns (with duplicates: a turn shared by two facts
    counts twice, matching the budget accounting).
    """
    # turn_id -> Turn view, and turn_id -> list of facts anchored to it.
    turn_view: dict[str, Turn] = {}
    turn_headers: dict[str, list[_FactWithTurns]] = defaultdict(list)
    standalone: list[_FactWithTurns] = []
    n_turns_anchored = 0

    for f, brings_turns in selected:
        if brings_turns and f.turns:
            for t in f.turns:
                turn_view[t.turn_id] = t
                turn_headers[t.turn_id].append(f)
                n_turns_anchored += 1
        else:
            standalone.append(f)

    # Stack headers above each turn in selection-rank order (highest first).
    blocks: list[tuple[tuple, str]] = []
    for tid, hdr_facts in turn_headers.items():
        hdr_facts_sorted = sorted(hdr_facts, key=lambda f: rank_index[f.fact_id])
        header = "\n".join(f"[fact: {f.text}]" for f in hdr_facts_sorted)
        t = turn_view[tid]
        blocks.append((t.sort_key, f"{header}\n{t.text}"))
    for f in standalone:
        blocks.append((f.sort_key, f"[fact: {f.text}]"))

    blocks.sort(key=lambda b: b[0])
    text = "\n\n".join(b[1] for b in blocks)

    rendered_turn_ids = [
        t.turn_id for t in sorted(turn_view.values(), key=lambda t: t.sort_key)
    ]
    return text, rendered_turn_ids, n_turns_anchored


# ---- variant b: facts as index only -----------------------------------------


class ActiveGraphSemIndexSystem(_FactEmbeddingMixin, ActiveGraphSemExtractSystem):
    """``activegraph-sem-index`` — facts as retrieval signal only.

    The cleanest ablation against det-embedding: facts are scored
    (embedding) and greedily selected, then REPLACED by their ``mentions``
    source turn(s). The reader sees only turn text, in the same
    ``"\\n\\n"``-joined chronological format det-embedding emits.

    Budget accounting (DESIGN CHOICE):
        Since facts never appear in the rendered context, the entire
        2,500-token budget is spent on turns. We walk facts in score
        order, follow ``mentions``, and add each not-yet-seen turn
        (cost = ``len(turn.text) + 2`` for the ``"\\n\\n"`` joiner, exactly
        as assemble() charges turns). A turn that would exceed the budget
        is skipped and ``truncated`` is set, but we keep walking lower-
        ranked facts whose turns may still fit — mirroring assemble()'s
        greedy "continue, don't break" packing so the only difference from
        det-embedding is WHICH turns are proposed (fact-driven vs
        turn-embedding-driven), not how the budget is packed.

        NOTE: no 1-hop ``precedes`` temporal expansion here (unlike
        det-embedding). The ``mentions`` edges already carry the turns the
        extractor judged to establish each fact — that IS the structural
        graph use for this variant.
    """

    name = "activegraph-sem-index"

    def __init__(
        self,
        *,
        embedding_model: str = "text-embedding-3-small",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        self._embedder: EmbeddingClient | None = None

    def retrieve(
        self, state: _SemState, question: str, question_date: str
    ) -> AssembledContext:
        facts = _facts_with_turns(state.state)
        scores = self._score(state, question, facts)
        ranked = _rank_facts(facts, scores)

        budget = self.token_budget
        running = 0
        truncated = False
        selected_turns: dict[str, Turn] = {}
        n_facts_with_evidence = 0

        for f in ranked:
            contributed = False
            for t in f.turns:
                if t.turn_id in selected_turns:
                    contributed = True
                    continue
                cost = _tok_count(t.text) + 2
                if running + cost > budget:
                    truncated = True
                    continue
                running += cost
                selected_turns[t.turn_id] = t
                contributed = True
            if contributed:
                n_facts_with_evidence += 1

        ordered = sorted(selected_turns.values(), key=lambda t: t.sort_key)
        text = "\n\n".join(t.text for t in ordered)
        rendered_turn_ids = [t.turn_id for t in ordered]

        # selected_fact_ids = facts whose evidence (>=1 mention turn)
        # actually reached the reader, in score order.
        selected_fact_ids = [
            f.fact_id
            for f in ranked
            if any(t.turn_id in selected_turns for t in f.turns)
        ]
        # anchored = total mention links across those facts (pre-dedup view).
        n_turns_anchored = sum(
            len(f.turns)
            for f in ranked
            if any(t.turn_id in selected_turns for t in f.turns)
        )

        meta = {
            **state.meta,
            "assembly_variant": "index",
            "retrieval_signal": "embedding",
            "embedding_model": self.embedding_model,
            "token_budget": self.token_budget,
            "n_facts_selected": len(selected_fact_ids),
            "n_facts_scored": len(facts),
            "n_facts_with_fitting_evidence": n_facts_with_evidence,
            "n_turns_anchored": n_turns_anchored,
            "n_unique_turns_rendered": len(rendered_turn_ids),
            "selected_turn_ids": rendered_turn_ids,
            "selected_fact_ids": selected_fact_ids,
        }
        return AssembledContext(text=text, truncated=truncated, meta=meta)
