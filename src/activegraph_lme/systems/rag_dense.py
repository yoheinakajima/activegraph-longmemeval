from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
from openai import OpenAI

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


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _embed_batch(model: str, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    # OpenAI batches up to a generous size; we keep it modest for memory.
    out = []
    B = 96
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        from activegraph_lme.activegraph.retrieve import truncate_for_embedding
        chunk = [truncate_for_embedding(t) for t in chunk]
        resp = _get_client().embeddings.create(model=model, input=chunk)
        out.extend([np.asarray(d.embedding, dtype=np.float32) for d in resp.data])
    arr = np.stack(out, axis=0)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


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
