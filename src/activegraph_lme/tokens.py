"""Token counting (cross-system yardstick only; authoritative reader counts
come from the Anthropic API `usage` field).

tiktoken downloads its BPE on first use. Importing this module never forces
the network hit. If the download fails (offline sandbox, blocked CDN), we
fall back to a ``len(text) / 4`` approximation and emit a one-time warning;
the authoritative reader counts in ``manifest.json`` are unaffected.
"""

from __future__ import annotations

import logging
import os

import tiktoken


log = logging.getLogger(__name__)

_ENC = None
_FALLBACK = False


def _get_enc():
    global _ENC, _FALLBACK
    if _ENC is not None or _FALLBACK:
        return _ENC
    try:
        _ENC = tiktoken.get_encoding("o200k_base")
    except Exception as e:  # offline / 403 from blob CDN, etc.
        _FALLBACK = True
        log.warning(
            "tiktoken could not load o200k_base (%s); falling back to "
            "char/4 token approximation for `context_tokens` only. "
            "Authoritative prompt/completion counts from the reader API "
            "are unaffected.",
            type(e).__name__,
        )
    return _ENC


def count_tokens(text: str) -> int:
    enc = _get_enc()
    if enc is None:
        return max(1, len(text) // 4) if text else 0
    return len(enc.encode(text, disallowed_special=()))


# Test hook: force fallback explicitly without touching the network.
if os.environ.get("AGLME_FORCE_TIKTOKEN_FALLBACK"):
    _FALLBACK = True
