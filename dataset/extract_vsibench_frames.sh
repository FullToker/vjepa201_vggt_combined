#!/usr/bin/env bash
# Wrapper around extract_vsibench_frames.py + make_vsibench_manifest.py.
#
# Usage:
#   bash dataset/extract_vsibench_frames.sh [VSI_DIR] [MANIFEST_OUT] [WORKERS]
set -euo pipefail

VSI_DIR="${1:-./source_data/vsibench}"
MANIFEST_OUT="${2:-./data/vsibench_manifest.jsonl}"
WORKERS="${3:-4}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Extracting FPS-sampled frames from: ${VSI_DIR}"
python3 "$SCRIPT_DIR/extract_vsibench_frames.py" \
  --vsi-dir "$VSI_DIR" \
  --fps 1.0 \
  --num-frames 8 \
  --mode mc \
  --workers "$WORKERS"

echo "==> Building manifest -> ${MANIFEST_OUT}"
python3 "$SCRIPT_DIR/make_vsibench_manifest.py" \
  --vsi-dir "$VSI_DIR" \
  --out "$MANIFEST_OUT" \
  --num-frames 8 \
  --mode mc

echo "==> Done."
