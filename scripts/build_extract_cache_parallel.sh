#!/usr/bin/env bash
# Fan out scripts/build_extract_cache.py across N worker processes that
# all append to the same JSONL cache file. Cross-process safety relies
# on fcntl.flock around each append (see _PersistentExtractionCache.put);
# the manifest is only stamped once by this parent at the end.
#
# Usage:
#   bash scripts/build_extract_cache_parallel.sh A           # 8 workers (default)
#   bash scripts/build_extract_cache_parallel.sh A 12        # 12 workers
set -euo pipefail

SEED="${1:-A}"
N="${2:-8}"

mkdir -p .scratch_build
LOG_DIR=".scratch_build/seed-${SEED}-workers"
rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

echo "Launching $N workers for seed $SEED ..."
pids=()
for i in $(seq 0 $((N-1))); do
  ACTIVEGRAPH_SEM_EXTRACT_CACHE_NO_FLUSH=1 \
    uv run python scripts/build_extract_cache.py \
      --seed "$SEED" --shard-of "${i}/${N}" \
      > "$LOG_DIR/worker-${i}.log" 2>&1 &
  pids+=($!)
done
echo "Worker pids: ${pids[*]}"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "WORKER PID $pid FAILED — see $LOG_DIR/"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "One or more workers failed; not stamping manifest."
  exit 1
fi

echo
echo "All workers done. Stamping final manifest + CHECKSUMS..."
uv run python -c "
from pathlib import Path
from activegraph_lme.config import load_config
from activegraph_lme.systems import build_system
cfg = load_config('config/run.yaml')
sys = build_system('activegraph-sem-extract', cfg, extract_seed='${SEED}')
sys._cache.flush_manifest()
print('Final n_entries:', len(sys._cache))
print('Manifest path :', sys._cache.manifest_path)
"
echo "Worker logs in $LOG_DIR"
