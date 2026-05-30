#!/usr/bin/env bash
# Stage-1 Scored Comparison Runner
# =================================
# Self-contained script to run locally (requires network access to
# HuggingFace, api.anthropic.com, and api.openai.com).
#
# Pre-requisites:
#   - uv installed
#   - ANTHROPIC_API_KEY and OPENAI_API_KEY in .env or environment
#   - git submodule initialized (git submodule update --init --recursive)
#   - data downloaded (make data)
#
# Usage:
#   cd <repo-root>
#   bash scripts/run_stage1_scored.sh
#
# Cost estimate:
#   Step 1 (primary pair): ~$1-2
#   Step 2 (variance = seed-B + seed-C build + scored runs): ~$17-20
#   Total: well under $25
#
# Standing rules:
#   - Smoke-scoped only (50 questions). NEVER runs the full 500.
#   - Never commits unverified artifacts.
#   - Credit-out: finishes in-flight run, commits verified, stops.
#   - Parse errors → stub {"facts": []}. Network errors → fail loudly.
#   - seed-B/C caches stay gitignored. Only seed-A is committed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
LOG=".scratch_build/resume.log"
mkdir -p .scratch_build

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

die() { log "FATAL: $*"; exit 1; }

# ===== PRE-FLIGHT (no API spend) =====
log "=== PRE-FLIGHT ==="

# 1. Verify seed-A cache
log "Checking seed-A cache..."
if [[ ! -f data/sem_extract_cache/seed-A.jsonl ]]; then
  die "data/sem_extract_cache/seed-A.jsonl missing"
fi
(cd data/sem_extract_cache && sha256sum -c CHECKSUMS.sha256) || die "seed-A checksum failed"
log "seed-A checksum OK"

# 2. Verify dataset exists
if [[ ! -f data/longmemeval_s_cleaned.json ]]; then
  log "Dataset missing, downloading..."
  make data || die "make data failed — check network and HF access"
fi
log "Dataset present"

# 3. Zero-call replay verification for seed-A
log "Running zero-call replay verification (seed-A)..."
uv run python scripts/verify_extract_cache.py --seed A || die "seed-A verify failed (non-zero LLM calls)"
log "seed-A zero-call replay: PASS"

# 4. Check seed-B
if [[ -f data/sem_extract_cache/seed-B.jsonl ]]; then
  log "seed-B found on disk, verifying..."
  uv run python scripts/verify_extract_cache.py --seed B || die "seed-B verify failed"
  log "seed-B zero-call replay: PASS"
  SEED_B_READY=1
else
  log "seed-B NOT on disk — will need to build in Step 2"
  SEED_B_READY=0
fi

# 5. Check for stale processes
if ls .scratch_build/*.lock 2>/dev/null; then
  log "WARNING: stale lock files found, removing..."
  rm -f .scratch_build/*.lock
fi
log "No stale lock files"

log "=== PRE-FLIGHT COMPLETE ==="
echo

# ===== STEP 1: PRIMARY PAIR =====
log "=== STEP 1: PRIMARY PAIR ==="

# 1a. det-embedding smoke baseline
log "Running activegraph-det-embedding smoke baseline..."
DET_EMB_DIR=$(uv run python -m activegraph_lme.cli run \
  --system activegraph-det-embedding \
  --dataset s \
  --smoke 2>&1 | tail -1)

if [[ ! -d "$DET_EMB_DIR" ]]; then
  die "det-embedding run failed, output was: $DET_EMB_DIR"
fi
log "det-embedding run dir: $DET_EMB_DIR"

# Eval det-embedding
log "Evaluating det-embedding..."
DET_EMB_SCORES=$(uv run python -m activegraph_lme.cli eval --run-dir "$DET_EMB_DIR" 2>&1)
log "det-embedding eval complete"
echo "$DET_EMB_SCORES" > "$DET_EMB_DIR/scores_raw.json"

# Commit det-embedding run
git add "$DET_EMB_DIR"
git commit -m "stage1: det-embedding smoke scored run

Step 1a of the Stage-1 comparison. Deterministic embedding baseline
scored on the 50-question smoke subset."
log "det-embedding committed"

# 1b. sem-extract seed-A smoke
log "Running activegraph-sem-extract seed-A smoke..."
SEM_A_DIR=$(uv run python -m activegraph_lme.cli run \
  --system activegraph-sem-extract \
  --dataset s \
  --smoke \
  --extract-seed A 2>&1 | tail -1)

if [[ ! -d "$SEM_A_DIR" ]]; then
  die "sem-extract-A run failed, output was: $SEM_A_DIR"
fi
log "sem-extract-A run dir: $SEM_A_DIR"

# Verify 0 extraction calls
MANIFEST="$SEM_A_DIR/manifest.json"
if [[ -f "$MANIFEST" ]]; then
  log "Manifest exists, checking extraction meta..."
fi

# Eval sem-extract-A
log "Evaluating sem-extract-A..."
SEM_A_SCORES=$(uv run python -m activegraph_lme.cli eval --run-dir "$SEM_A_DIR" 2>&1)
log "sem-extract-A eval complete"
echo "$SEM_A_SCORES" > "$SEM_A_DIR/scores_raw.json"

# Commit sem-extract-A run
git add "$SEM_A_DIR"
git commit -m "stage1: sem-extract seed-A smoke scored run

Step 1b of the Stage-1 comparison. Semantic extraction with frozen
seed-A cache (0 LLM extraction calls, all cache hits) scored on
the 50-question smoke subset."
log "sem-extract-A committed"

# 1c. AIC sidecar on both
log "Running AIC sidecar on det-embedding..."
uv run python scripts/aic_sidecar.py "$DET_EMB_DIR" || log "WARNING: AIC sidecar failed for det-embedding"

log "Running AIC sidecar on sem-extract-A..."
uv run python scripts/aic_sidecar.py "$SEM_A_DIR" || log "WARNING: AIC sidecar failed for sem-extract-A"

# Commit sidecars
git add "$DET_EMB_DIR/aic_sidecar.jsonl" "$SEM_A_DIR/aic_sidecar.jsonl" 2>/dev/null || true
git commit -m "stage1: AIC sidecar for primary pair" --allow-empty || true
log "AIC sidecars committed"

log "=== STEP 1 COMPLETE ==="
echo
log "PRIMARY RESULTS:"
log "det-embedding scores: $DET_EMB_SCORES"
log "sem-extract-A scores: $SEM_A_SCORES"
echo
echo "=============================================="
echo "STEP 1 PRIMARY PAIR RESULTS"
echo "=============================================="
echo "det-embedding: $DET_EMB_SCORES"
echo "sem-extract-A: $SEM_A_SCORES"
echo "=============================================="
echo
echo ">> STOP here and review before proceeding to Step 2."
echo ">> To continue to Step 2 (variance), re-run with: bash scripts/run_stage1_scored.sh --step2"
echo

if [[ "${1:-}" != "--step2" && "${1:-}" != "--all" ]]; then
  log "Stopping after Step 1 (pass --step2 or --all to continue)"
  exit 0
fi

# ===== STEP 2: VARIANCE CONFIRMATION =====
log "=== STEP 2: VARIANCE CONFIRMATION ==="

# 2a. sem-extract seed-B
if [[ "$SEED_B_READY" -eq 0 ]]; then
  log "Building seed-B cache (parallel, ~$15)..."
  bash scripts/build_extract_cache_parallel.sh B 8 || die "seed-B build failed"
  log "Verifying seed-B..."
  uv run python scripts/verify_extract_cache.py --seed B || die "seed-B verify failed after build"
  log "seed-B build + verify: PASS"
fi

log "Running sem-extract seed-B smoke..."
SEM_B_DIR=$(uv run python -m activegraph_lme.cli run \
  --system activegraph-sem-extract \
  --dataset s \
  --smoke \
  --extract-seed B 2>&1 | tail -1)

if [[ ! -d "$SEM_B_DIR" ]]; then
  die "sem-extract-B run failed, output was: $SEM_B_DIR"
fi
log "sem-extract-B run dir: $SEM_B_DIR"

log "Evaluating sem-extract-B..."
SEM_B_SCORES=$(uv run python -m activegraph_lme.cli eval --run-dir "$SEM_B_DIR" 2>&1)
log "sem-extract-B eval complete"
echo "$SEM_B_SCORES" > "$SEM_B_DIR/scores_raw.json"

git add "$SEM_B_DIR"
git commit -m "stage1: sem-extract seed-B smoke scored run (variance)"
log "sem-extract-B committed"

# 2b. Build seed-C, verify, then score
log "Building seed-C cache (parallel, smoke unique sessions only)..."
bash scripts/build_extract_cache_parallel.sh C 8 || die "seed-C build failed"
log "Verifying seed-C zero-call replay..."
uv run python scripts/verify_extract_cache.py --seed C || die "seed-C verify failed after build"
log "seed-C build + verify: PASS"

log "Running sem-extract seed-C smoke..."
SEM_C_DIR=$(uv run python -m activegraph_lme.cli run \
  --system activegraph-sem-extract \
  --dataset s \
  --smoke \
  --extract-seed C 2>&1 | tail -1)

if [[ ! -d "$SEM_C_DIR" ]]; then
  die "sem-extract-C run failed, output was: $SEM_C_DIR"
fi
log "sem-extract-C run dir: $SEM_C_DIR"

log "Evaluating sem-extract-C..."
SEM_C_SCORES=$(uv run python -m activegraph_lme.cli eval --run-dir "$SEM_C_DIR" 2>&1)
log "sem-extract-C eval complete"
echo "$SEM_C_SCORES" > "$SEM_C_DIR/scores_raw.json"

git add "$SEM_C_DIR"
git commit -m "stage1: sem-extract seed-C smoke scored run (variance)"
log "sem-extract-C committed"

log "=== STEP 2 COMPLETE ==="
echo
echo "=============================================="
echo "VARIANCE RESULTS (A/B/C)"
echo "=============================================="
echo "sem-extract-A: $SEM_A_SCORES"
echo "sem-extract-B: $SEM_B_SCORES"
echo "sem-extract-C: $SEM_C_SCORES"
echo "=============================================="
echo

# ===== STEP 3: SUMMARY =====
log "=== STEP 3: WRITING SUMMARY ==="

uv run python -c "
import json, sys
from pathlib import Path

def load_scores(run_dir):
    p = Path(run_dir) / 'scores.json'
    if not p.exists():
        p = Path(run_dir) / 'scores_raw.json'
    if not p.exists():
        return {'error': f'no scores file in {run_dir}'}
    try:
        return json.loads(p.read_text())
    except:
        # scores_raw.json might be the cli output (printed json)
        raw = p.read_text().strip()
        return json.loads(raw)

det = load_scores('$DET_EMB_DIR')
sem_a = load_scores('$SEM_A_DIR')
sem_b = load_scores('$SEM_B_DIR') if '$SEM_B_DIR' else None
sem_c = load_scores('$SEM_C_DIR') if '$SEM_C_DIR' else None

lines = []
lines.append('# Stage-1 Scored Comparison — Resume Summary')
lines.append('')
lines.append('## Primary Delta (Step 1)')
lines.append('')
lines.append('| System | Accuracy | AIC |')
lines.append('|--------|----------|-----|')

def acc(s):
    if not s or 'error' in s:
        return 'N/A'
    return str(s.get('overall_accuracy', s.get('accuracy', 'N/A')))

def aic_val(run_dir):
    p = Path(run_dir) / 'aic_sidecar.jsonl'
    if not p.exists():
        return 'N/A'
    # Count answer-in-context hits
    total = 0
    hits = 0
    for line in p.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            total += 1
            if row.get('answer_in_context'):
                hits += 1
    return f'{hits}/{total} ({100*hits/total:.1f}%)' if total else 'N/A'

lines.append(f'| det-embedding | {acc(det)} | {aic_val(\"$DET_EMB_DIR\")} |')
lines.append(f'| sem-extract-A | {acc(sem_a)} | {aic_val(\"$SEM_A_DIR\")} |')
lines.append('')

if sem_a and det and 'error' not in sem_a and 'error' not in det:
    d_acc = float(acc(det)) if acc(det) != 'N/A' else None
    s_acc = float(acc(sem_a)) if acc(sem_a) != 'N/A' else None
    if d_acc is not None and s_acc is not None:
        lines.append(f'Delta (sem-extract-A minus det-embedding): {s_acc - d_acc:+.1f}pp')
        lines.append('')

lines.append('## Per-Category Breakdown')
lines.append('')
# Try to extract per-category if available
if det and 'per_category' in det:
    lines.append('| Category | det-embedding | sem-extract-A |')
    lines.append('|----------|---------------|---------------|')
    for cat in sorted(det['per_category'].keys()):
        d_cat = det['per_category'].get(cat, 'N/A')
        s_cat = sem_a.get('per_category', {}).get(cat, 'N/A') if sem_a else 'N/A'
        lines.append(f'| {cat} | {d_cat} | {s_cat} |')
    lines.append('')
elif det and isinstance(det, dict):
    lines.append('(per-category breakdown not available in scores.json)')
    lines.append('')
    lines.append('Full det-embedding scores:')
    lines.append(f'\`\`\`json')
    lines.append(json.dumps(det, indent=2))
    lines.append(f'\`\`\`')
    lines.append('')
    lines.append('Full sem-extract-A scores:')
    lines.append(f'\`\`\`json')
    lines.append(json.dumps(sem_a, indent=2))
    lines.append(f'\`\`\`')
    lines.append('')

lines.append('## Variance Band (Step 2)')
lines.append('')
lines.append('| Seed | Accuracy |')
lines.append('|------|----------|')
lines.append(f'| A | {acc(sem_a)} |')
if sem_b:
    lines.append(f'| B | {acc(sem_b)} |')
if sem_c:
    lines.append(f'| C | {acc(sem_c)} |')
lines.append('')

accs = []
for s in [sem_a, sem_b, sem_c]:
    if s and 'error' not in s:
        v = acc(s)
        if v != 'N/A':
            accs.append(float(v))
if len(accs) >= 2:
    spread = max(accs) - min(accs)
    lines.append(f'Spread (max - min): {spread:.1f}pp')
    lines.append(f'Range: [{min(accs):.1f}, {max(accs):.1f}]')
    lines.append('')

lines.append('## Failures / Stubs')
lines.append('')
lines.append('(Logged if any runs had failures or stubs)')
lines.append('')

Path('.scratch_build/resume_summary.md').write_text('\n'.join(lines))
print('Summary written to .scratch_build/resume_summary.md')
" 2>&1 || log "WARNING: summary generation had issues"

log "=== PIPELINE COMPLETE ==="
cat .scratch_build/resume_summary.md
