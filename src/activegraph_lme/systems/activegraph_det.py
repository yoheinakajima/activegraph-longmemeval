"""Deterministic ActiveGraph system adapter (Mode A).

Two sub-variants, selected by ``retrieval_signal``:
  * ``lexical``   — IDF-weighted distinctive-token overlap
  * ``embedding`` — pinned ``text-embedding-3-small`` cosine similarity

Both share the same graph build (Turn nodes + temporal + co-occurrence
edges, all deterministic from raw text) and the same budgeted assembly.
The token budget mirrors the turn-level RAG baselines so the comparison
is not confounded by context size.

NO LLM is called at ingest or retrieve time. The Anthropic reader is
called by the harness on the assembled context, exactly as for every
other system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..activegraph.graph import IngestState, build_graph
from ..activegraph.retrieve import (
    AssemblyResult,
    EmbeddingClient,
    assemble,
    score_embedding,
    score_lexical,
)
from ..data import LMEInstance
from .base import AssembledContext


log = logging.getLogger(__name__)


RetrievalSignal = Literal["lexical", "embedding"]


@dataclass
class _State:
    state: IngestState
    # Embedding mode caches the pinned per-turn embeddings here so retrieve()
    # is cheap and deterministic across re-calls.
    turn_embeddings: np.ndarray | None = None
    # Stats logged into the per-question manifest record via meta.
    meta: dict[str, Any] = field(default_factory=dict)


class ActiveGraphDetSystem:
    """Deterministic ActiveGraph; both sub-variants behind one adapter."""

    def __init__(
        self,
        retrieval_signal: RetrievalSignal,
        *,
        token_budget: int,
        min_token_length: int,
        min_session_cooccurrence: int,
        max_doc_freq_fraction: float,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        if retrieval_signal not in ("lexical", "embedding"):
            raise ValueError(
                f"retrieval_signal must be 'lexical' or 'embedding', got {retrieval_signal!r}"
            )
        self.retrieval_signal = retrieval_signal
        self.token_budget = token_budget
        self.min_token_length = min_token_length
        self.min_session_cooccurrence = min_session_cooccurrence
        self.max_doc_freq_fraction = max_doc_freq_fraction
        self.embedding_model = embedding_model
        self._embedder: EmbeddingClient | None = None

    @property
    def name(self) -> str:
        return f"activegraph-det-{self.retrieval_signal}"

    def _get_embedder(self) -> EmbeddingClient:
        if self._embedder is None:
            self._embedder = EmbeddingClient(model=self.embedding_model)
        return self._embedder

    # ---- frozen System protocol ----

    def ingest(self, instance: LMEInstance) -> _State:
        ingest_state = build_graph(
            instance.haystack_session_ids,
            instance.haystack_dates,
            instance.haystack_sessions,
            min_token_length=self.min_token_length,
            min_session_cooccurrence=self.min_session_cooccurrence,
            max_doc_freq_fraction=self.max_doc_freq_fraction,
        )
        state = _State(state=ingest_state, meta=dict(ingest_state.stats()))
        if self.retrieval_signal == "embedding":
            embedder = self._get_embedder()
            texts = [t.text for t in ingest_state.turns]
            state.turn_embeddings = embedder.embed(texts)
        return state

    def retrieve(
        self, state: _State, question: str, question_date: str
    ) -> AssembledContext:
        if self.retrieval_signal == "lexical":
            scores = score_lexical(
                state.state, question, min_token_length=self.min_token_length
            )
        else:
            embedder = self._get_embedder()
            scores, _ = score_embedding(
                state.state, question, embedder, turn_embeddings=state.turn_embeddings
            )

        res: AssemblyResult = assemble(
            state.state, scores, token_budget=self.token_budget
        )

        meta = {
            **state.meta,
            "n_selected_turns": len(res.selected_turn_ids),
            "n_seeds": res.n_seeds,
            "n_temporal_expansions": res.n_expanded,
            "retrieval_signal": self.retrieval_signal,
            "token_budget": self.token_budget,
        }
        return AssembledContext(text=res.text, truncated=res.truncated, meta=meta)
