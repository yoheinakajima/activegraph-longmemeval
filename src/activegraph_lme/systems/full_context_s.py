from __future__ import annotations

from dataclasses import dataclass

from ..data import LMEInstance
from ..tokens import count_tokens as _count
from .base import AssembledContext, format_session


@dataclass
class _State:
    sessions: list[str]   # already-formatted, in chronological (oldest-first) order
    token_budget: int


class FullContextS:
    """Stuff the entire haystack (chronological), truncating oldest sessions
    first if the configured token budget is exceeded.

    The 115k figure in the upstream paper is measured with the Llama tokenizer;
    actual token counts under Sonnet's tokenizer differ. The harness records
    real per-instance token counts in the manifest.
    """

    name = "full-context-s"

    def __init__(self, token_budget: int = 180_000) -> None:
        self.token_budget = token_budget

    def ingest(self, instance: LMEInstance) -> _State:
        formatted = [
            format_session(sid, date, turns)
            for sid, date, turns in zip(
                instance.haystack_session_ids,
                instance.haystack_dates,
                instance.haystack_sessions,
            )
        ]
        return _State(sessions=formatted, token_budget=self.token_budget)

    def retrieve(self, state: _State, question: str, question_date: str) -> AssembledContext:
        # Drop from the oldest end until we fit.
        kept = list(state.sessions)
        text = "\n\n".join(kept)
        truncated = False
        while kept and _count(text) > state.token_budget:
            kept.pop(0)
            truncated = True
            text = "\n\n".join(kept)
        return AssembledContext(text=text, truncated=truncated)
