from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..data import LMEInstance


@dataclass
class AssembledContext:
    text: str
    truncated: bool = False     # True iff a budget forced dropping content
    # Optional extras a system may surface for debugging.
    meta: dict[str, Any] | None = None


class System(Protocol):
    """Common interface across all five systems.

    Lifecycle per question:
        state = system.ingest(instance)
        ctx = system.retrieve(state, instance.question, instance.question_date)
        # answer = reader.generate(SYSTEM_PROMPT, format_user(ctx, question))

    `retrieve` MUST be deterministic given the same `state` + question.
    The harness asserts this once per system via a repeat-call check.
    """

    name: str

    def ingest(self, instance: LMEInstance) -> Any: ...
    def retrieve(self, state: Any, question: str, question_date: str) -> AssembledContext: ...


# ---- shared session formatting -----------------------------------------------

def format_session(session_id: str, date: str, turns: list[dict[str, Any]]) -> str:
    """Stable textual rendering of a session, used by every system."""
    lines = [f"### Session {session_id} ({date})"]
    for t in turns:
        role = t.get("role", "?")
        content = t.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
