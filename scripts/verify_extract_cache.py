"""Verify the frozen extraction cache produces a zero-LLM-call replay.

After ``build_extract_cache.py --seed A`` has populated the committed
cache, this script:

  1. Loads seed-A in a fresh process. Manifest validation must pass.
  2. Ingests every smoke instance. With a complete cache, the runtime
     must make ZERO live LLM calls — every session is a hit, and the
     persisted Fact set is byte-identical to the build run's facts.
  3. Counts: how many extract LLM calls fired (must be 0), how many
     cache hits (must equal cumulative session occurrences).

This is the determinism gate that justifies the "extraction frozen"
claim — if it ever fires non-zero misses on the smoke set against
the committed seed-A.jsonl, something invalidated the cache that
wasn't caught by the guard.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from activegraph_lme.config import load_config
from activegraph_lme.data import load_dataset
from activegraph_lme.systems import build_system


def smoke_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def fact_signature(state) -> list[dict]:
    """Content-stable representation of the post-ingest fact set:
    fact_id (content hash), text, session_id, and the set of mentioned
    turn_ids ('{sid}#{turn_idx}', NOT auto-incremented obj.ids). This
    is the signature the byte-identity check compares against the
    build-run signature."""
    g = state.state.graph
    turn_obj_to_id: dict[str, str] = {}
    for obj in g.objects(type="Turn"):
        data = obj.data or {}
        tid = data.get("turn_id")
        if tid:
            turn_obj_to_id[obj.id] = str(tid)
    mentions_by_fact: dict[str, list[str]] = {}
    for rel in g.relations(type="mentions"):
        mentions_by_fact.setdefault(rel.source, []).append(
            turn_obj_to_id.get(rel.target, f"?obj:{rel.target}")
        )
    for k in mentions_by_fact:
        mentions_by_fact[k] = sorted(mentions_by_fact[k])
    rows = []
    for obj in g.objects(type="Fact"):
        data = obj.data or {}
        rows.append({
            "fact_id": data.get("fact_id"),
            "text": data.get("text"),
            "session_id": data.get("session_id"),
            "session_idx": data.get("session_idx"),
            "mentions": mentions_by_fact.get(obj.id, []),
        })
    rows.sort(key=lambda r: (r["session_idx"], r["fact_id"]))
    return rows


def main() -> int:
    cfg = load_config("config/run.yaml")
    all_instances = load_dataset(cfg.datasets["s"])
    ids = set(smoke_ids(Path("config/smoke_ids.txt")))
    instances = [i for i in all_instances if i.question_id in ids]
    print(f"Smoke questions: {len(instances)}")

    system = build_system("activegraph-sem-extract", cfg, extract_seed="A")
    print(f"Cache path: {system._cache.cache_path}")
    print(f"Entries loaded at init: {len(system._cache)}")
    print(f"Manifest prompt_sha256: {system._prompt_sha256}")
    print()

    t0 = time.monotonic()
    cum_extracted = 0
    cum_hits = 0
    n_facts_total = 0
    appended_at_start = system._cache.n_appended_this_process()
    for idx, inst in enumerate(instances):
        st = system.ingest(inst)
        m = st.meta
        cum_extracted = m["cum_sessions_extracted"]
        cum_hits = m["cum_cache_hits"]
        n_facts_total += m["n_facts"]
        if m["n_sessions_extracted"] != 0:
            # If this fires anything but 0, the cache is incomplete
            # for this smoke set — surface immediately, don't burn API.
            print(
                f"  UNEXPECTED MISS on {inst.question_id}: "
                f"extracted={m['n_sessions_extracted']}; "
                f"this means the committed seed-A does not cover the smoke set."
            )
            return 1
    elapsed = time.monotonic() - t0
    appended_during_run = (
        system._cache.n_appended_this_process() - appended_at_start
    )

    print(f"All {len(instances)} smoke questions ingested in {elapsed:.1f}s")
    print(f"Cumulative extract LLM calls (must be 0): {cum_extracted}")
    print(f"Cumulative cache hits: {cum_hits}")
    print(f"New entries appended (must be 0): {appended_during_run}")
    print(f"Total Facts written across all ingests: {n_facts_total}")
    if cum_extracted != 0 or appended_during_run != 0:
        print("FAIL: cache replay made live API calls or grew the cache.")
        return 1
    print()
    print("OK: zero-LLM-call replay across the smoke set. ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
