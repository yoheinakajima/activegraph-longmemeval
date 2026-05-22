from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ReaderOutput:
    text: str
    prompt_tokens: int        # authoritative, from API usage
    completion_tokens: int    # authoritative, from API usage
    resolved_model: str       # exact dated snapshot returned by the API


class Reader(Protocol):
    """Tool-free, web-free LLM reader.

    Implementations MUST assert at construction time that no tools/web are
    enabled and MUST NOT pass `tools=` or any web/browse parameter on calls.
    """

    model: str  # requested alias

    def generate(self, system: str, user: str) -> ReaderOutput: ...
