#!/usr/bin/env bash
# Fetches the two LongMemEval cleaned dataset files into data/, then either
# verifies SHA256s against data/CHECKSUMS.sha256 (if present) or records them
# on first run. Fails non-zero on mismatch.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data
cd data

BASE="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
FILES=(longmemeval_oracle.json longmemeval_s_cleaned.json)

for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo ">> downloading $f"
    curl -fL --retry 4 --retry-delay 2 -o "$f" "$BASE/$f"
  else
    echo ">> $f already present, skipping download"
  fi
done

if [[ -f CHECKSUMS.sha256 && -s CHECKSUMS.sha256 ]]; then
  echo ">> verifying checksums against data/CHECKSUMS.sha256"
  sha256sum -c CHECKSUMS.sha256
else
  echo ">> CHECKSUMS.sha256 missing or empty; recording current SHA256s (first-run mode)"
  sha256sum "${FILES[@]}" > CHECKSUMS.sha256
  cat CHECKSUMS.sha256
  echo
  echo ">> committed CHECKSUMS.sha256 will be the authoritative pin from now on."
  echo ">> COMMIT THIS FILE."
fi
