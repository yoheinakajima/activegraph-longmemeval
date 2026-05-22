"""One-shot probe: send a minimal Anthropic call with the configured alias
and report what the API returned as ``response.model``.

If the resolved string equals the requested alias (rather than a dated
snapshot like ``...-YYYYMMDD``), the paper's "pinned model" claim is weak —
we should hardcode the dated snapshot into config/run.yaml.

Usage:
    uv run python scripts/check_resolved_model.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from activegraph_lme.config import load_config
from activegraph_lme.reader import AnthropicReader


def main() -> int:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY missing; set it in .env first.", file=sys.stderr)
        return 2

    cfg = load_config()
    requested = cfg.reader.model
    reader = AnthropicReader(
        model=requested,
        temperature=cfg.reader.temperature,
        max_tokens=16,
    )
    out = reader.generate(
        system="You are a helpful assistant.",
        user="Reply with exactly: ok",
    )

    resolved = out.resolved_model
    is_dated = bool(resolved) and any(ch.isdigit() for ch in resolved.split("-")[-1])
    print(f"requested:        {requested}")
    print(f"resolved (API):   {resolved}")
    print(f"prompt_tokens:    {out.prompt_tokens}")
    print(f"completion_tokens:{out.completion_tokens}")
    print(f"response_text:    {out.text!r}")

    if not resolved:
        print("\nFAIL: API returned no model string. Refuse to ship runs against this.")
        return 3
    if resolved == requested or not is_dated:
        print(
            "\nWARN: API returned the bare alias rather than a dated snapshot. "
            "For paper-grade pinning, hardcode the dated form in config/run.yaml "
            "(reader.model) and re-run."
        )
        return 1

    print(
        "\nOK: resolved to a dated snapshot. Manifests written by `aglme run` "
        "will pin exactly this string under `reader_model_resolved`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
