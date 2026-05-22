"""ActiveGraph adapter — STUB.

This file defines the FROZEN adapter interface that the round-two real
ActiveGraph implementation will plug into. The current implementation is a
deliberately trivial baseline (chronological recency under a token budget) so
the full harness runs end-to-end. It is NOT a real ActiveGraph run.

TODO(round 2): replace `ingest()` with real graph construction and
`retrieve()` with the real query path. The interface below must not change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..data import LMEInstance
from ..tokens import count_tokens as _count
from .base import AssembledContext, format_session


log = logging.getLogger(__name__)


@dataclass(frozen=True)  # frozen → equality-by-value, ensures determinism check is meaningful
class GraphState:
    """Opaque state produced by ingest(). The round-two impl will store the
    real graph; for now we store the ordered formatted sessions only.
    """

    sessions: tuple[str, ...]
    token_budget: int


class ActiveGraphSystem:
    """STUB. See module docstring."""

    name = "activegraph"

    def __init__(self, token_budget: int = 180_000) -> None:
        self.token_budget = token_budget
        log.warning(
            "ActiveGraphSystem is a STUB (recency-with-budget). "
            "Replace before reporting real ActiveGraph numbers."
        )

    # ---- frozen interface (do not change signatures in round 2) ----

    def ingest(self, instance: LMEInstance) -> GraphState:
        sessions = tuple(
            format_session(sid, date, turns)
            for sid, date, turns in zip(
                instance.haystack_session_ids,
                instance.haystack_dates,
                instance.haystack_sessions,
            )
        )
        return GraphState(sessions=sessions, token_budget=self.token_budget)

    def retrieve(self, state: GraphState, question: str, question_date: str) -> AssembledContext:
        # STUB policy: most-recent-first under budget.
        kept: list[str] = []
        running = 0
        truncated = False
        for s in reversed(state.sessions):
            n = _count(s) + 2  # +2 approximates the "\n\n" joiner
            if running + n > state.token_budget:
                truncated = True
                break
            kept.append(s)
            running += n
        kept.reverse()
        return AssembledContext(text="\n\n".join(kept), truncated=truncated)
