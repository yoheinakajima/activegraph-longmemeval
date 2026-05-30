"""Offline, $0 verification of the build_extract_cache write path.

Drives the REAL serial loop (`_run_units` + `_persist_role` +
`_PersistentExtractionCache.put`) with a MOCKED extractor — no graph
build, no Anthropic API, no spend. Proves:

  (a) end-to-end write path: 10 sessions x 2 roles -> 20 entries land in
      seed-A-v2.jsonl within 5 seconds, each entry flushed per-call.
  (b) no partner-role gate / no holes: a forced parse-error on session
      3's assistant extraction still leaves every other unit on disk AND
      a stubbed entry for 3/assistant — 20 entries, no gaps.

Run: python scripts/verify_build_write_path.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_extract_cache as bm  # noqa: E402
from activegraph_lme.systems.activegraph_sem_extract import (  # noqa: E402
    _PersistentExtractionCache,
    _compute_combined_prompt_sha256,
)

PASS, FAIL = "OK", "FAIL"


def _mk_sessions(n: int):
    """n synthetic session tuples (sid, sdate, s_idx, turns)."""
    return [
        (f"sess-{i:02d}", "2025-01-01", i, [{"role": "user", "content": "hi"}])
        for i in range(n)
    ]


def _fresh_cache(d: Path) -> _PersistentExtractionCache:
    return _PersistentExtractionCache(
        cache_dir=d,
        seed="A-v2",
        prompt_sha256=_compute_combined_prompt_sha256(),
        extractor_model_requested="claude-sonnet-4-5",
    )


def _read_entries(path: Path):
    """Return list of (session_id, role, n_facts) for every JSONL line."""
    out = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        out.append(
            (obj["session_id"], obj["role"], len(obj["parsed"]["facts"]))
        )
    return out


def _mock_canned(s, role):
    """Always succeeds with 1-2 canned facts. No API."""
    sid = s[0]
    facts = [{"text": f"{role} fact A for {sid}", "mentioned_turn_idxs": [0]}]
    if role == "assistant":
        facts.append({"text": f"{role} fact B for {sid}", "mentioned_turn_idxs": []})
    return sid, f"csum-{sid}", {"facts": facts}, "claude-canned-snapshot"


def _mock_with_parse_error(s, role):
    """Like _mock_canned but session 3's assistant extraction parse-errors
    (parsed_dump is None) — exercises the stub-on-failure path."""
    sid = s[0]
    if sid == "sess-03" and role == "assistant":
        return sid, f"csum-{sid}", None, None  # parse-error: no parsed dump
    return _mock_canned(s, role)


def _units(sessions):
    return [(s, role) for s in sessions for role in ("user", "assistant")]


def check_a() -> tuple[str, str]:
    sessions = _mk_sessions(10)
    with tempfile.TemporaryDirectory() as d:
        cache = _fresh_cache(Path(d))
        t0 = time.monotonic()
        n = bm._run_units(
            cache, _units(sessions), serial=True, workers=1,
            extract_fn=_mock_canned,
        )
        elapsed = time.monotonic() - t0
        entries = _read_entries(cache.cache_path)
        roles = {(sid, role) for sid, role, _ in entries}
        expected = {(f"sess-{i:02d}", r) for i in range(10)
                    for r in ("user", "assistant")}
    ok = (n == 20 and len(entries) == 20 and roles == expected
          and elapsed < 5.0)
    return (PASS if ok else FAIL,
            f"(a) write path: appended={n} on_disk={len(entries)} "
            f"unique={len(roles)} elapsed={elapsed:.3f}s (<5s)")


def check_b() -> tuple[str, str]:
    sessions = _mk_sessions(10)
    with tempfile.TemporaryDirectory() as d:
        cache = _fresh_cache(Path(d))
        n = bm._run_units(
            cache, _units(sessions), serial=True, workers=1,
            extract_fn=_mock_with_parse_error,
        )
        entries = _read_entries(cache.cache_path)
        by_key = {(sid, role): nf for sid, role, nf in entries}
        roles = set(by_key)
        expected = {(f"sess-{i:02d}", r) for i in range(10)
                    for r in ("user", "assistant")}
        # The parse-errored unit is present as a stub (0 facts); every
        # OTHER unit landed normally (>=1 fact) — no holes, no blocking.
        stub_ok = by_key.get(("sess-03", "assistant")) == 0
        others_nonempty = all(
            nf >= 1 for (sid, role), nf in by_key.items()
            if not (sid == "sess-03" and role == "assistant")
        )
    ok = (n == 20 and len(entries) == 20 and roles == expected
          and stub_ok and others_nonempty)
    return (PASS if ok else FAIL,
            f"(b) no-hole / no-gate: on_disk={len(entries)} "
            f"parse_err_stub_facts={by_key.get(('sess-03','assistant'))} "
            f"others_nonempty={others_nonempty}")


def main() -> int:
    results = [check_a(), check_b()]
    n_fail = 0
    for status, msg in results:
        print(f"  [{status}] {msg}")
        if status == FAIL:
            n_fail += 1
    print()
    if n_fail:
        print(f"{n_fail} failure(s)")
        return 1
    print("write-path verification passed ($0, no API)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
