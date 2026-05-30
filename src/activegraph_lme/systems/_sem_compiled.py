"""Shared scaffolding for the two *compiled semantic memory* ActiveGraph
systems (Stage-1 follow-up to ``activegraph-sem-extract``).

Motivation (the Borges-regression / "compile, don't replace" argument)
    Stage-1's ``activegraph-sem-extract`` scored a *pool of Facts + Turns*
    lexically and let facts crowd the raw user turns out of the reader
    context. It lost 16pp of smoke accuracy to ``det-embedding`` —
    concentrated on single-session-user and multi-session, exactly the
    categories where the discarded raw user turn *was* the answer. The
    lesson: semantic memory should COMPILE from the log (be an index INTO
    the turns), not REPLACE the log.

    Both systems here keep the seed-A extraction cache and the Fact→Turn
    ``mentions`` provenance edges. They reuse the *same retrieval signal*
    (embedding cosine over the Fact pool — facts are the index), and the
    *same* 2,500-token budget as ``det-embedding`` for parity. They differ
    only in how :meth:`retrieve` assembles the reader context:

      * ``activegraph-sem-hybrid`` (variant a): facts as labeled headers
        ANCHORED above their provenance turns.
      * ``activegraph-sem-index``  (variant b): facts are retrieval signal
        ONLY; the reader sees just the turns (det-embedding's exact
        format), selected via facts.

Zero extraction LLM calls
    Ingest is delegated verbatim to :class:`ActiveGraphSemExtractSystem`,
    which loads the committed seed-A cache. With the smoke set fully
    cached every session is a cache HIT, so ``run_until_idle`` fires no
    ``@llm_behavior`` and ``meta.n_sessions_extracted == 0``. The only
    live API cost is the OpenAI *embedding* of fact texts at retrieve time
    (same model — ``text-embedding-3-small`` — det-embedding uses), and
    the Anthropic reader call the harness makes on the assembled context.

Scoring vs. assembly seam
    ``_score_facts`` (embedding cosine, impure: hits OpenAI) is kept
    separate from each subclass's ``_assemble`` (pure: deterministic given
    the fact scores). This lets the assembly invariants — dedup,
    chronological order, budget — be unit-tested offline with injected
    scores, no API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..activegraph.graph import IngestState
from ..activegraph.retrieve import EmbeddingClient
from .activegraph_sem_extract import ActiveGraphSemExtractSystem, _SemState


# Facts sort AFTER any plausible turn_idx in the same session — mirrors the
# offset retrieve.py::_FACT_SEQ_OFFSET uses so a fact emitted from a session
# sorts immediately after that session's turns in a chronological pass.
_FACT_SEQ_OFFSET = 10**9


@dataclass(frozen=True)
class _FactView:
    """Projection of one ``Fact`` object plus its resolved provenance turns.

    ``turn_ids`` are the stable ``{session_id}#{turn_idx}`` ids of the
    turns the fact ``mentions``, already de-duplicated and sorted
    chronologically by the source turn's ``sort_key``. A fact whose
    extraction recorded no supporting turn (``mentioned_turn_idxs == []``)
    has an empty ``turn_ids`` — it has no anchor.
    """

    fact_id: str
    obj_id: str
    text: str
    sort_key: tuple
    turn_ids: tuple[str, ...]


def project_facts(state: IngestState) -> list[_FactView]:
    """Project every ``Fact`` object + its ``mentions`` turns into
    :class:`_FactView`s, in deterministic package-insertion order.

    The ``mentions`` edges were written ``source=fact_obj.id ->
    target=turn_obj.id`` (see activegraph_sem_extract._write_facts_to_graph),
    so we group targets by source. Returns ``[]`` when no Fact objects
    exist (keeps the systems inert on a turn-only graph).
    """
    mentions_by_fact_obj: dict[str, list[str]] = {}
    for r in state.graph.relations(type="mentions"):
        mentions_by_fact_obj.setdefault(r.source, []).append(r.target)

    out: list[_FactView] = []
    for obj in state.graph.objects(type="Fact"):
        data = obj.data or {}
        text = str(data.get("text", ""))
        session_date = str(data.get("session_date", ""))
        session_idx = int(data.get("session_idx", 0))
        try:
            seq = int(obj.id.rsplit("#", 1)[1])
        except (IndexError, ValueError):
            seq = 0
        sort_key = (session_date, session_idx, _FACT_SEQ_OFFSET + seq)
        fact_id = str(data.get("fact_id") or obj.id)

        # Resolve provenance turn object-ids -> chronological turn_ids,
        # de-duplicated (a fact may list the same turn twice).
        turn_views = []
        seen: set[str] = set()
        for tobj_id in mentions_by_fact_obj.get(obj.id, ()):
            tv = state.by_object_id.get(tobj_id)
            if tv is None or tv.turn_id in seen:
                continue
            seen.add(tv.turn_id)
            turn_views.append(tv)
        turn_views.sort(key=lambda tv: tv.sort_key)

        out.append(
            _FactView(
                fact_id=fact_id,
                obj_id=obj.id,
                text=text,
                sort_key=sort_key,
                turn_ids=tuple(tv.turn_id for tv in turn_views),
            )
        )
    return out


class _SemCompiledBase:
    """Common ingest + fact-scoring for the two compiled-memory systems.

    Subclasses implement :meth:`retrieve` (and a pure ``_assemble``).
    """

    name: str  # set by subclasses

    def __init__(
        self,
        *,
        token_budget: int,
        min_token_length: int,
        min_session_cooccurrence: int,
        max_doc_freq_fraction: float,
        extractor_model: str = "claude-sonnet-4-5",
        extract_seed: str = "A",
        embedding_model: str = "text-embedding-3-small",
        extraction_cache_dir: str | Path | None = None,
    ) -> None:
        self.token_budget = token_budget
        self.min_token_length = min_token_length
        self.embedding_model = embedding_model
        # Ingest is delegated wholesale to the sem-extract system so the
        # cache, the Fact writes and the `mentions` edges are byte-identical
        # to that system's. We add NO behaviors and change NO extraction
        # path — only assembly differs.
        self._extractor = ActiveGraphSemExtractSystem(
            token_budget=token_budget,
            min_token_length=min_token_length,
            min_session_cooccurrence=min_session_cooccurrence,
            max_doc_freq_fraction=max_doc_freq_fraction,
            extractor_model=extractor_model,
            extract_seed=extract_seed,
            extraction_cache_dir=extraction_cache_dir,
        )
        self._embedder: EmbeddingClient | None = None

    # ---- ingest (delegated, zero extraction LLM calls under seed-A) --------

    def ingest(self, instance) -> _SemState:
        return self._extractor.ingest(instance)

    # ---- fact scoring (embedding cosine — the shared retrieval signal) -----

    def _get_embedder(self) -> EmbeddingClient:
        if self._embedder is None:
            self._embedder = EmbeddingClient(model=self.embedding_model)
        return self._embedder

    def _score_facts(
        self, question: str, facts: list[_FactView]
    ) -> dict[str, float]:
        """Cosine similarity between the question and each fact's text.

        ``EmbeddingClient`` L2-normalizes every vector and is
        content-addressed, so re-scoring the same (question, facts) is
        byte-identical — the harness's repeat-call determinism check holds
        without us caching anything on the state. Facts are the retrieval
        index for BOTH variants; this is the only place the signal is
        computed.
        """
        if not facts:
            return {}
        embedder = self._get_embedder()
        fact_vecs = embedder.embed([f.text for f in facts])
        q_vec = embedder.embed([question])[0]
        raw = fact_vecs @ q_vec
        return {f.fact_id: float(raw[i]) for i, f in enumerate(facts)}

    @staticmethod
    def _rank_facts(
        facts: list[_FactView], scores: dict[str, float]
    ) -> list[_FactView]:
        """Greedy ranking order: score desc, chronological (sort_key) tie-break.

        Total order on ``sort_key`` makes the ranking — and therefore the
        whole assembly — deterministic even when two facts tie on cosine.
        """
        return sorted(
            facts,
            key=lambda f: (-scores.get(f.fact_id, 0.0), f.sort_key),
        )
