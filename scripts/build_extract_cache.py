"""Build a frozen extraction cache for activegraph-sem-extract.

Run extraction over the union of the smoke subset's haystacks. Each
unique session_id+content_sha256 is extracted at most once; the
parsed _ExtractedFactList is appended to
``data/sem_extract_cache/seed-{seed}.jsonl`` and stamped into
``seed-{seed}.manifest.json`` + ``CHECKSUMS.sha256``.

Cost-controlled: re-runs are free because the in-process load-on-init
+ write-through cache shortcircuits already-extracted sessions. Only
seed-A is the committed canonical artifact (per the experiment design);
seed-B/C are gitignored variance samples.

Usage:
  uv run python scripts/build_extract_cache.py --seed A
  uv run python scripts/build_extract_cache.py --seed B   # variance sample
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", choices=["A", "B", "C"], default="A")
    ap.add_argument("--config", default="config/run.yaml")
    ap.add_argument(
        "--smoke-ids", default="config/smoke_ids.txt",
        help="Path to the frozen smoke question_id list.",
    )
    ap.add_argument(
        "--shard-of", type=str, default="",
        help=(
            "Worker-shard spec 'i/N' (e.g. '3/8'). When set, the script "
            "processes only the i-th of N strided slices of smoke "
            "instances (smoke[i::N]) and skips the final manifest flush. "
            "The parent (no --shard-of) writes the manifest after all "
            "workers exit. Used by scripts/build_extract_cache_parallel.sh "
            "to fan out the build."
        ),
    )
    args = ap.parse_args()

    shard_idx = -1
    shard_total = 1
    is_worker = bool(args.shard_of)
    if is_worker:
        try:
            i_str, n_str = args.shard_of.split("/")
            shard_idx = int(i_str)
            shard_total = int(n_str)
            assert 0 <= shard_idx < shard_total
        except Exception as e:
            raise SystemExit(f"--shard-of expects 'i/N' (0<=i<N), got {args.shard_of!r}: {e}")

    cfg = load_config(args.config)
    all_instances = load_dataset(cfg.datasets["s"])
    ids = set(smoke_ids(Path(args.smoke_ids)))
    instances = [i for i in all_instances if i.question_id in ids]
    if len(instances) != len(ids):
        raise SystemExit(
            f"smoke set references question_ids not in dataset; "
            f"want={len(ids)} got={len(instances)}"
        )
    if is_worker:
        instances = instances[shard_idx::shard_total]
        print(f"Worker shard {shard_idx}/{shard_total}: {len(instances)} questions")
    unique_sessions = {
        sid for i in instances for sid in i.haystack_session_ids
    }
    print(f"Seed: {args.seed}")
    print(f"Smoke questions: {len(instances)}")
    print(f"Total haystack occurrences: "
          f"{sum(len(i.haystack_session_ids) for i in instances)}")
    print(f"Unique sessions to cache: {len(unique_sessions)}")

    system = build_system("activegraph-sem-extract", cfg, extract_seed=args.seed)
    print(f"Cache path: {system._cache.cache_path}")
    print(f"Entries already loaded: {len(system._cache)}")
    print(f"Prompt sha256: {system._prompt_sha256}")
    print()

    t0 = time.monotonic()
    n_appended_before = system._cache.n_appended_this_process()
    cum_extracted = 0
    cum_hits = 0
    for idx, inst in enumerate(instances):
        ti = time.monotonic()
        st = system.ingest(inst)
        m = st.meta
        cum_extracted = m["cum_sessions_extracted"]
        cum_hits = m["cum_cache_hits"]
        elapsed = time.monotonic() - ti
        print(
            f"  [{idx+1:>2}/{len(instances)}] {inst.question_id} "
            f"sessions={m['n_sessions_total']:>2} "
            f"extracted={m['n_sessions_extracted']:>2} "
            f"hits={m['n_cache_hits']:>2}  "
            f"cum: extracted={cum_extracted:>4} hits={cum_hits:>3}  "
            f"({elapsed:.1f}s)"
        )

    total_elapsed = time.monotonic() - t0
    print()
    print(f"Done in {total_elapsed:.1f}s")
    n_appended = system._cache.n_appended_this_process() - n_appended_before
    print(f"New entries appended this run: {n_appended}")
    print(f"Total entries in cache: {len(system._cache)}")
    print(f"Cumulative extracted (LLM calls): {cum_extracted}")
    print(f"Cumulative cache hits: {cum_hits}")
    if not is_worker:
        # Re-stamp manifest in case the last ingest had no misses.
        system._cache.flush_manifest()
        with open(system._cache.manifest_path) as f:
            manifest = json.load(f)
        print(f"\nManifest:\n{json.dumps(manifest, indent=2, sort_keys=True)}")
    else:
        # Worker mode: parent re-stamps the manifest after all workers exit.
        # Workers only append to the JSONL (cross-process safe via fcntl
        # locking in _PersistentExtractionCache.put).
        print(f"Worker {shard_idx}/{shard_total} done; parent will stamp manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
