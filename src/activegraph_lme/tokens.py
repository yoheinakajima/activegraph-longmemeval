"""Token counting (cross-system yardstick for `context_tokens`).

The authoritative prompt/completion counts come from the reader API's
``usage`` field and are always recorded with source ``"api"``. This module
covers only the *context* token count.

Source resolution:
  * ``tiktoken``   — authoritative; on-disk BPE loaded by tiktoken.
  * ``charfallback`` — ``len(text) // 4`` approximation. Recorded only when
    tiktoken cannot load (no network on first run AND no cache populated).

To keep tiktoken offline-friendly we set ``TIKTOKEN_CACHE_DIR`` to a
project-local path on import, so the BPE is downloaded ONCE (any
network-enabled run) and then cached for all subsequent runs — including
subprocess children spawned by the matrix orchestrator.

Paper runs MUST verify ``manifest["context_token_source"] == "tiktoken"``.
The harness enforces this via ``--require-authoritative-tokens``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import tiktoken


log = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _REPO_ROOT / ".tiktoken_cache"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_DEFAULT_CACHE))
_DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)


Source = Literal["tiktoken", "charfallback"]


_ENC = None
_SOURCE: Source | None = None
_WARNED = False


def _force_fallback() -> bool:
    return os.environ.get("AGLME_FORCE_TIKTOKEN_FALLBACK") == "1"


def _get_enc():
    global _ENC, _SOURCE, _WARNED
    if _SOURCE is not None:
        return _ENC
    if _force_fallback():
        _SOURCE = "charfallback"
        if not _WARNED:
            log.warning("AGLME_FORCE_TIKTOKEN_FALLBACK=1 set; using char/4 approximation.")
            _WARNED = True
        return None
    try:
        _ENC = tiktoken.get_encoding("o200k_base")
        _SOURCE = "tiktoken"
    except Exception as e:
        _SOURCE = "charfallback"
        if not _WARNED:
            log.warning(
                "tiktoken could not load o200k_base (%s); falling back to "
                "char/4 approximation for `context_tokens`. Paper runs with "
                "--require-authoritative-tokens will fail until tiktoken can "
                "download its BPE once (set TIKTOKEN_CACHE_DIR to persist).",
                type(e).__name__,
            )
            _WARNED = True
    return _ENC


def count_tokens(text: str) -> int:
    enc = _get_enc()
    if enc is None:
        return max(1, len(text) // 4) if text else 0
    return len(enc.encode(text, disallowed_special=()))


def token_source() -> Source:
    """The source that will be / has been used for ``count_tokens``."""
    _get_enc()
    assert _SOURCE is not None
    return _SOURCE
