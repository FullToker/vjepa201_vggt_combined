#!/usr/bin/env bash
# Wrapper around convert_openeqa_manifest.py -- pose-FPS-samples 8 frames per
# question (hm3d-v0/* from downloaded states pickles, scannet-v0/* from
# posed_images/*.txt) and builds a fusion_gv inference manifest.
#
# Usage:
#   bash dataset/convert_openeqa_manifest.sh [OPENEQA_DIR] [MANIFEST_OUT] [NUM_FRAMES] [SCANNET_DIR]
set -euo pipefail

OPENEQA_DIR="${1:-./source_data/openeqa}"
MANIFEST_OUT="${2:-./data/openeqa_manifest.jsonl}"
NUM_FRAMES="${3:-8}"
SCANNET_DIR="${4:-./source_data/scannet}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Building OpenEQA manifest (pose-FPS ${NUM_FRAMES}-frame sampling) -> ${MANIFEST_OUT}"
python3 "$SCRIPT_DIR/convert_openeqa_manifest.py" \
  --openeqa-dir "$OPENEQA_DIR" \
  --scannet-dir "$SCANNET_DIR" \
  --out "$MANIFEST_OUT" \
  --num-frames "$NUM_FRAMES"

echo "==> Done."
