#!/usr/bin/env bash
# Wrapper around extract_vsibench_frames.py + make_vsibench_manifest.py.
#
# Usage:
#   bash dataset/extract_vsibench_frames.sh [VSI_DIR] [MANIFEST_OUT] [WORKERS] [NUM_FRAMES] [POOL_SIZE]
#
# NUM_FRAMES / POOL_SIZE default to 8 / 64 (unchanged prior behaviour). Frames
# land in <VSI_DIR>/frames{NUM_FRAMES}_fps/, so different counts never collide.
# For the 32-frame VSI-Bench eval see dataset/extract_vsibench_frames_32f.sbatch.
set -euo pipefail

VSI_DIR="${1:-./source_data/vsibench}"
MANIFEST_OUT="${2:-./data/vsibench_manifest.jsonl}"
WORKERS="${3:-4}"
NUM_FRAMES="${4:-8}"
POOL_SIZE="${5:-64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Extracting FPS(farthest-point)-sampled frames from: ${VSI_DIR}  (num_frames=${NUM_FRAMES}, pool=${POOL_SIZE})"
python3 "$SCRIPT_DIR/extract_vsibench_frames.py" \
  --vsi-dir "$VSI_DIR" \
  --num-frames "$NUM_FRAMES" \
  --pool-size "$POOL_SIZE" \
  --mode mc \
  --workers "$WORKERS"

echo "==> Building manifest -> ${MANIFEST_OUT}"
python3 "$SCRIPT_DIR/make_vsibench_manifest.py" \
  --vsi-dir "$VSI_DIR" \
  --out "$MANIFEST_OUT" \
  --num-frames "$NUM_FRAMES" \
  --mode mc

echo "==> Done."
