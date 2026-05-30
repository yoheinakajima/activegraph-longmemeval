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

Parallelism
    --workers N runs the (independent, LLM-bound) per-session extractions
    across N processes; the parent serially writes the returned results
    into the JSONL + manifest. No cross-process file locking needed
    because only the parent writes.
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


def _extract_one_session(
    args: tuple[str, str, int, list[dict]],
) -> tuple[str, str, dict, str | None]:
    """Worker: extract BOTH roles' facts for ONE session.

    Returns (sid, csum, {role: parsed_dump_or_None}, resolved_model).
    Builds a throwaway single-session graph + runtime so each worker is
    fully independent (no shared graph mutation across processes). Both
    extractor behaviors react to the single emitted extract_request.
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
        behaviors=[_sem_extract_handler, _sem_extract_handler_assistant],
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
        print(f"  [warn] extraction failed for {sid}: {e!r}", file=sys.stderr)
        return sid, csum, {"user": None, "assistant": None}, None

    results: dict[str, dict | None] = {"user": None, "assistant": None}
    resolved: str | None = None
    for ev in graph.events:
        if ev.type != "llm.responded":
            continue
        role = _BEHAVIOR_ROLE.get(ev.payload.get("behavior"))
        if role is None:
            continue
        p = ev.payload.get("parsed")
        if p is not None:
            results[role] = p
        if ev.payload.get("model"):
            resolved = str(ev.payload["model"])
    return sid, csum, results, resolved


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


def _persist(cache: _PersistentExtractionCache, sid, csum, results, resolved) -> int:
    n = 0
    for role, dump in results.items():
        if dump is None:
            continue
        cache.put(
            sid, csum, role,
            _ExtractedFactList.model_validate(dump),
            extractor_model_resolved=resolved,
        )
        n += 1
    return n


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", default="A-v2")
    ap.add_argument("--dataset", default="s")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke-ids", default="config/smoke_ids.txt")
    ap.add_argument("--config", default="config/run.yaml")
    args = ap.parse_args()

    cfg = load_config()
    instances = load_dataset(Path(cfg.datasets[args.dataset]))
    if args.smoke:
        instances = _filter_smoke(instances, Path(args.smoke_ids))

    cache = _PersistentExtractionCache(
        cache_dir=Path("data/sem_extract_cache"),
        seed=args.seed,
        prompt_sha256=_compute_combined_prompt_sha256(),
        extractor_model_requested="claude-sonnet-4-5",
    )

    sessions = list(_iter_sessions(instances))
    if args.resume:
        # A session is done only when BOTH roles are already cached.
        def _done(s) -> bool:
            csum = _csum_for_session(s[0], s[1], s[3])
            return (cache.get(s[0], csum, "user") is not None
                    and cache.get(s[0], csum, "assistant") is not None)
        sessions = [s for s in sessions if not _done(s)]

    print(f"[build] {len(sessions)} sessions to extract (seed={args.seed}, "
          f"workers={args.workers}, 2 roles each)")

    n_written = 0
    if args.workers <= 1:
        for s in sessions:
            sid, csum, results, resolved = _extract_one_session(s)
            n_written += _persist(cache, sid, csum, results, resolved)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_extract_one_session, s): s for s in sessions}
            for fut in as_completed(futs):
                sid, csum, results, resolved = fut.result()
                n_written += _persist(cache, sid, csum, results, resolved)

    cache.flush_manifest()
    print(f"[build] wrote {n_written} new entries to {cache.cache_path}")
    print(f"[build] manifest: {cache.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
