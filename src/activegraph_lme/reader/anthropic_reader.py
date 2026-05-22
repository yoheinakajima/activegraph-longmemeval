from __future__ import annotations

import os

from anthropic import Anthropic

from .base import ReaderOutput


class AnthropicReader:
    """Claude reader, pinned to a dated Sonnet snapshot at temperature=0.

    Hard guarantees enforced here:
      - NO tools are ever passed (no `tools=` kwarg on create()).
      - NO web access / browsing parameter.
      - NO server-side tool use is requested.

    The model alias requested in config (e.g. ``claude-sonnet-4-5``) is
    sent as-is; the dated snapshot the API actually served is captured
    from the response (``response.model``) and surfaced as
    ``ReaderOutput.resolved_model`` so the manifest can pin it.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: str | None = None,
        *,
        tools: object = None,  # accept-and-reject so callers can't sneak tools in
    ) -> None:
        if tools is not None:
            raise AssertionError(
                "AnthropicReader is tool-free by design; remove `tools=`."
            )
        if temperature != 0.0:
            raise AssertionError(
                "Determinism requires temperature=0.0; got "
                f"{temperature!r}."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def generate(self, system: str, user: str) -> ReaderOutput:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            # Intentionally NO tools, NO extra_headers enabling betas, NO web.
        )

        resolved = getattr(resp, "model", None)
        if not resolved:
            raise RuntimeError(
                "Anthropic response missing `model` field; cannot pin reader "
                "snapshot. Refusing to record an ambiguous run."
            )

        parts = []
        for block in resp.content:
            # Defensive: text blocks only. Tool-use blocks should never appear
            # because we never enabled tools.
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
            else:
                raise RuntimeError(
                    f"Unexpected non-text content block from reader: {block!r}"
                )
        text = "".join(parts).strip()

        return ReaderOutput(
            text=text,
            prompt_tokens=int(resp.usage.input_tokens),
            completion_tokens=int(resp.usage.output_tokens),
            resolved_model=resolved,
        )
