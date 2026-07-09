from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..data import LMEInstance
from .base import AssembledContext, format_session


@dataclass
class _Doc:
    sort_key: tuple
    text: str


@dataclass
class _State:
    docs: list[_Doc]
    embeddings: np.ndarray  # (N, D), L2-normalized
    top_k: int


_embedders: dict[str, object] = {}


def _embed_batch(model: str, texts: list[str]) -> np.ndarray:
    from activegraph_lme.activegraph.retrieve import EmbeddingClient

    embedder = _embedders.get(model)
    if embedder is None:
        embedder = EmbeddingClient(model=model)
        _embedders[model] = embedder
    return embedder.embed(texts)  # type: ignore[attr-defined]


class RagDense:
    """Dense embedding retrieval (OpenAI text-embedding-3-small)."""

    name = "rag-dense"

    def __init__(
        self,
        granularity: Literal["turn", "session"] = "session",
        top_k: int = 10,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.granularity = granularity
        self.top_k = top_k
        self.embedding_model = embedding_model

    def ingest(self, instance: LMEInstance) -> _State:
        docs: list[_Doc] = []
        for s_idx, (sid, date, turns) in enumerate(
            zip(
                instance.haystack_session_ids,
                instance.haystack_dates,
                instance.haystack_sessions,
            )
        ):
            if self.granularity == "session":
                docs.append(
                    _Doc(sort_key=(date, s_idx, 0), text=format_session(sid, date, turns))
                )
            else:
                for t_idx, turn in enumerate(turns):
                    role = turn.get("role", "?")
                    content = turn.get("content", "")
                    docs.append(
                        _Doc(
                            sort_key=(date, s_idx, t_idx),
                            text=f"[Session {sid} ({date})] {role}: {content}",
                        )
                    )

        texts = [d.text for d in docs]
        emb = _embed_batch(self.embedding_model, texts)
        return _State(docs=docs, embeddings=emb, top_k=self.top_k)

    def retrieve(self, state: _State, question: str, question_date: str) -> AssembledContext:
        if not state.docs:
            return AssembledContext(text="", truncated=False)
        q_emb = _embed_batch(self.embedding_model, [question])[0]
        sims = state.embeddings @ q_emb  # cosine since both are L2-normalized
        ranked = sorted(
            range(len(state.docs)),
            key=lambda i: (-float(sims[i]), state.docs[i].sort_key),
        )
        picked = ranked[: state.top_k]
        picked.sort(key=lambda i: state.docs[i].sort_key)
        text = "\n\n".join(state.docs[i].text for i in picked)
        return AssembledContext(text=text, truncated=False)
