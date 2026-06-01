"""Build (or extend) a persistent seed extraction cache for the
sem-extract family WITHOUT running the scored benchmark.

Why this exists
    The scored run path (`cli run --system activegraph-sem-extract ...`)
    will transparently populate the cache on a miss, but that couples
    cache construction to a full reader+judge spend. This script lets us
    build the frozen seed cache as a standalone, parallelizable step —
    then commit it as the canonical experiment input.

    Role-aware (v2): each session is extracted TWICE — once for
    user-authored facts, once for assistant-authored facts — so a single
    session yields two cache entries keyed by
    (session_id, content_sha256, role). The cache manifest is pinned to
    the COMBINED prompt signature (both templates), which is what
    invalidates the user-only seed-A under the new behavior set.

    Cost note: extraction is the only LLM spend here (no reader, no
    judge). v2 does ~2x the calls of the original single-role build.

Usage:
    python scripts/build_extract_cache.py --seed A-v2 --dataset s --smoke
    python scripts/build_extract_cache.py --seed A-v2 --dataset s --smoke --workers 8

Parallelism / write model
    Each (session, role) extraction is an independent unit of work. The
    parent persists each returned result to the JSONL **immediately**, as
    soon as it completes — writes are never gated on the partner role (or
    any other session) finishing. A user-fact extraction that succeeds
    while the assistant-fact extraction for the same session is stuck in a
    retry loop still lands on disk right away.

    --serial (or --workers <= 1) runs a plain Python `for` loop over
    (session, role) pairs: no ProcessPoolExecutor, no concurrent.futures,
    nothing between the extraction and the write but a function return.
    This is the recommended, deterministic path and isolates the write
    path from any executor quirks.

    --workers N (N > 1, and not --serial) fans the per-(session, role)
    units across N processes; the parent still does every write, one entry
    at a time, as futures complete.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from activegraph_lme.config import load_config
from activegraph_lme.data import load_dataset, LMEInstance
from activegraph_lme.systems.activegraph_sem_extract import (
    _ExtractedFactList,
    _PersistentExtractionCache,
    _build_llm_provider,
    _compute_combined_prompt_sha256,
    _sem_extract_handler,
    _sem_extract_handler_assistant,
    _BEHAVIOR_ROLE,
    _EXTRACT_REQUEST_TYPE,
)
import activegraph as ag


def _content_sha256(session_text: str) -> str:
    return hashlib.sha256(session_text.encode("utf-8")).hexdigest()


def _session_text(turn_views) -> str:
    return "\n".join(
        f"[turn {tv.turn_idx}] {tv.role}: {tv.content}" for tv in turn_views
    )


def _filter_smoke(instances, smoke_ids_path: Path):
    ids = {
        line.strip()
        for line in smoke_ids_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    return [i for i in instances if i.question_id in ids]


_ROLE_BEHAVIORS = {
    "user": _sem_extract_handler,
    "assistant": _sem_extract_handler_assistant,
}


def _extract_role_for_session(
    args: tuple[str, str, int, list[dict]],
    role: str,
) -> tuple[str, str, dict | None, str | None]:
    """Worker: extract ONE role's facts for ONE session.

    Returns (sid, csum, parsed_dump_or_None, resolved_model). ``None`` for
    the parsed dump means the extraction failed or the LLM output did not
    parse (a parse-error); the caller persists an empty stub in that case
    so the entry still lands on disk and never blocks the partner role.

    Builds a throwaway single-session graph + runtime with ONLY this
    role's behavior registered, so the two roles of a session are fully
    independent units of work: a wedged assistant extraction can never
    stall the user write (or vice versa), and each unit can be persisted
    the instant it returns.
    """
    sid, sdate, s_idx, session_turns = args

    from activegraph_lme.activegraph.graph import build_graph

    state = build_graph(
        [sid], [sdate], [session_turns],
        min_token_length=4, min_session_cooccurrence=2, max_doc_freq_fraction=0.5,
    )
    turn_views = state.turns
    session_text = _session_text(turn_views)
    csum = _content_sha256(session_text)
    graph = state.graph

    runtime = ag.Runtime(
        graph,
        behaviors=[_ROLE_BEHAVIORS[role]],
        llm_provider=_build_llm_provider("claude-sonnet-4-5"),
        seed=0,
    )
    payload = {
        "session_id": sid,
        "session_date": sdate,
        "session_idx": s_idx,
        "session_text": session_text,
        "turn_object_ids": [tv.object_id for tv in turn_views],
        "n_turns": len(turn_views),
    }
    graph.emit(
        ag.Event(
            id=graph.ids.event(),
            type=_EXTRACT_REQUEST_TYPE,
            payload=payload,
            actor="build_extract_cache",
        )
    )
    try:
        runtime.run_until_idle()
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] {role} extraction failed for {sid}: {e!r}", file=sys.stderr)
        return sid, csum, None, None

    parsed_dump: dict | None = None
    resolved: str | None = None
    for ev in graph.events:
        if ev.type != "llm.responded":
            continue
        if _BEHAVIOR_ROLE.get(ev.payload.get("behavior")) != role:
            continue
        p = ev.payload.get("parsed")
        if p is not None:
            parsed_dump = p
        if ev.payload.get("model"):
            resolved = str(ev.payload["model"])
    return sid, csum, parsed_dump, resolved


def _iter_sessions(instances: list[LMEInstance]):
    """Yield (sid, sdate, s_idx, session_turns) for every UNIQUE session id
    across all instances (haystack sessions repeat across questions)."""
    seen: set[str] = set()
    for inst in instances:
        for s_idx, (sid, sdate, sess) in enumerate(
            zip(inst.haystack_session_ids, inst.haystack_dates, inst.haystack_sessions)
        ):
            if sid in seen:
                continue
            seen.add(sid)
            yield sid, sdate, s_idx, sess


def _csum_for_session(sid, sdate, sess) -> str:
    from activegraph_lme.activegraph.graph import build_graph

    state = build_graph(
        [sid], [sdate], [sess],
        min_token_length=4, min_session_cooccurrence=2, max_doc_freq_fraction=0.5,
    )
    return _content_sha256(_session_text(state.turns))


def _persist_role(
    cache: _PersistentExtractionCache, sid, csum, role, parsed_dump, resolved
) -> int:
    """Write ONE (session, role) result through to disk immediately.

    On a parse-error / failed extraction (``parsed_dump is None``) we still
    persist an empty fact-list stub: the entry lands on disk, ``--resume``
    won't re-attempt it, and — critically — it never leaves a hole that a
    stuck partner role could block behind. ``cache.put`` is the
    write-through append (flush + fsync per entry) and is idempotent, so a
    re-run is a no-op for already-present keys.

    Returns the number of entries actually appended to the JSONL (0 if the
    key was already present), measured from the cache's append counter.
    """
    parsed = (
        _ExtractedFactList(facts=[])
        if parsed_dump is None
        else _ExtractedFactList.model_validate(parsed_dump)
    )
    before = cache._n_appended_this_process
    cache.put(sid, csum, role, parsed, extractor_model_resolved=resolved)
    return cache._n_appended_this_process - before


def _clear_stale_locks(cache_dir: Path) -> None:
    """Remove leftover lock / partial / scratch artifacts from a prior
    killed build so they can't wedge or mislead a fresh run.

    The cache append uses an in-fd ``fcntl.flock`` (no on-disk lock file),
    but a previously-aborted run or external tooling may have left
    ``*.lock`` / ``*.partial`` sidecars or a ``.scratch_build/`` dir behind.
    We don't assume they're absent — sweep them defensively.
    """
    patterns = ["*.lock", "*.partial"]
    for d in (cache_dir, cache_dir / ".scratch_build", Path(".scratch_build")):
        if not d.exists():
            continue
        for pat in patterns:
            for p in d.glob(pat):
                try:
                    p.unlink()
                    print(f"[build] cleared stale lock artifact: {p}")
                except OSError as e:  # noqa: BLE001
                    print(f"  [warn] could not remove {p}: {e!r}", file=sys.stderr)


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", default="A-v2")
    ap.add_argument("--dataset", default="s")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--serial",
        action="store_true",
        help="Force the pure for-loop path (no ProcessPoolExecutor) "
             "regardless of --workers. Implied by --workers <= 1.",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke-ids", default="config/smoke_ids.txt")
    ap.add_argument("--config", default="config/run.yaml")
    args = ap.parse_args()

    cfg = load_config()
    instances = load_dataset(Path(cfg.datasets[args.dataset]))
    if args.smoke:
        instances = _filter_smoke(instances, Path(args.smoke_ids))

    cache_dir = Path("data/sem_extract_cache")
    _clear_stale_locks(cache_dir)

    cache = _PersistentExtractionCache(
        cache_dir=cache_dir,
        seed=args.seed,
        prompt_sha256=_compute_combined_prompt_sha256(),
        extractor_model_requested="claude-sonnet-4-5",
    )

    sessions = list(_iter_sessions(instances))

    # Unit of work = (session, role). Each is extracted and persisted
    # independently so neither role blocks the other.
    units: list[tuple[tuple[str, str, int, list[dict]], str]] = [
        (s, role) for s in sessions for role in ("user", "assistant")
    ]
    if args.resume:
        # Resume is per-role now: skip only the (session, role) pairs that
        # are already on disk, so a session with just its user fact cached
        # still gets its assistant fact attempted.
        def _cached(unit) -> bool:
            s, role = unit
            csum = _csum_for_session(s[0], s[1], s[3])
            return cache.get(s[0], csum, role) is not None
        units = [u for u in units if not _cached(u)]

    serial = args.serial or args.workers <= 1
    print(f"[build] {len(sessions)} sessions / {len(units)} (session,role) units "
          f"to extract (seed={args.seed}, "
          f"mode={'serial' if serial else f'parallel x{args.workers}'})")

    n_written = _run_units(cache, units, serial=serial, workers=args.workers)

    cache.flush_manifest()
    print(f"[build] wrote {n_written} new entries to {cache.cache_path}")
    print(f"[build] manifest: {cache.manifest_path}")
    return 0


def _run_units(
    cache: _PersistentExtractionCache,
    units,
    *,
    serial: bool,
    workers: int,
    extract_fn=_extract_role_for_session,
) -> int:
    """Extract + persist each (session, role) unit, returning the count of
    new JSONL entries appended.

    Every result is persisted the moment it is produced — there is no
    batching and no partner-role gate — so a wedged unit can never hold
    back a completed one. ``extract_fn`` is injectable so offline tests can
    drive the exact same loop with a mocked extractor (no API spend).
    """
    n_written = 0
    if serial:
        # Pure for-loop: nothing between extraction and the write-through
        # append but a function return. No ProcessPoolExecutor.
        for s, role in units:
            sid, csum, parsed_dump, resolved = extract_fn(s, role)
            n_written += _persist_role(cache, sid, csum, role, parsed_dump, resolved)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(extract_fn, s, role): role
                for (s, role) in units
            }
            for fut in as_completed(futs):
                role = futs[fut]
                sid, csum, parsed_dump, resolved = fut.result()
                n_written += _persist_role(cache, sid, csum, role, parsed_dump, resolved)
    return n_written


if __name__ == "__main__":
    raise SystemExit(main())
