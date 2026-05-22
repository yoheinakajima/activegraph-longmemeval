from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rank_bm25 import BM25Okapi

from ..data import LMEInstance
from .base import AssembledContext, format_session


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass
class _Doc:
    sort_key: tuple   # (date_str, idx) → deterministic tie-break
    text: str         # the textual chunk to retrieve and surface


@dataclass
class _State:
    docs: list[_Doc]
    bm25: BM25Okapi
    top_k: int


class RagBM25:
    """BM25 retrieval over either turn-level or session-level chunks.

    Determinism: BM25 ties broken by the stable per-doc sort_key
    (date, position) rather than by argsort instability.
    """

    name = "rag-bm25"

    def __init__(self, granularity: Literal["turn", "session"] = "session", top_k: int = 10) -> None:
        self.granularity = granularity
        self.top_k = top_k

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
                text = format_session(sid, date, turns)
                docs.append(_Doc(sort_key=(date, s_idx, 0), text=text))
            else:  # turn
                for t_idx, turn in enumerate(turns):
                    role = turn.get("role", "?")
                    content = turn.get("content", "")
                    text = f"[Session {sid} ({date})] {role}: {content}"
                    docs.append(_Doc(sort_key=(date, s_idx, t_idx), text=text))

        tokenized = [_tokenize(d.text) for d in docs]
        # rank_bm25 needs at least one token per doc; fall back to a placeholder.
        tokenized = [t if t else ["<empty>"] for t in tokenized]
        bm25 = BM25Okapi(tokenized)
        return _State(docs=docs, bm25=bm25, top_k=self.top_k)

    def retrieve(self, state: _State, question: str, question_date: str) -> AssembledContext:
        if not state.docs:
            return AssembledContext(text="", truncated=False)
        scores = state.bm25.get_scores(_tokenize(question))
        # Deterministic ordering: (-score, sort_key)
        ranked = sorted(
            range(len(state.docs)),
            key=lambda i: (-float(scores[i]), state.docs[i].sort_key),
        )
        picked = ranked[: state.top_k]
        # Present in chronological order so the reader sees a time-ordered context.
        picked.sort(key=lambda i: state.docs[i].sort_key)
        text = "\n\n".join(state.docs[i].text for i in picked)
        return AssembledContext(text=text, truncated=False)
