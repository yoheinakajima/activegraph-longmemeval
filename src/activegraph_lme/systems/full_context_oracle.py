from __future__ import annotations

from dataclasses import dataclass

from ..data import LMEInstance
from .base import AssembledContext, format_session


@dataclass
class _OracleState:
    text: str


class FullContextOracle:
    """Upper bound: feed only the sessions tagged as evidence."""

    name = "full-context-oracle"

    def ingest(self, instance: LMEInstance) -> _OracleState:
        evidence_ids = set(instance.answer_session_ids)
        chunks = []
        for sid, date, turns in zip(
            instance.haystack_session_ids,
            instance.haystack_dates,
            instance.haystack_sessions,
        ):
            if sid in evidence_ids:
                chunks.append(format_session(sid, date, turns))
        return _OracleState(text="\n\n".join(chunks))

    def retrieve(self, state: _OracleState, question: str, question_date: str) -> AssembledContext:
        return AssembledContext(text=state.text, truncated=False)
