#!/usr/bin/env bash
# Downloads OpenEQA's question-answer file, the open (non-gated) HM3D RGB
# episode frames, and the HM3D agent-state pickles (raw pose, no rendering
# needed) -- then builds the pose-FPS 8-frame manifest.
#
# ScanNet-sourced episodes (episode_history "scannet-v0/*") are NOT fetched
# here -- OpenEQA's own ScanNet path requires a signed data-use agreement
# plus a separate multi-hour local extraction pass. Instead,
# convert_openeqa_manifest.py reuses this repo's existing ScanNet
# posed-image data (dataset/download_scannet_posed.sh output) if present
# under --scannet-dir; otherwise scannet-v0/* rows are just skipped.
#
# Usage:
#   bash dataset/download_openeqa.sh [OUTPUT_DIR] [MANIFEST_OUT] [NUM_FRAMES] [SCANNET_DIR]
#
# Dataset page: https://github.com/facebookresearch/open-eqa
set -euo pipefail

QA_URL="https://raw.githubusercontent.com/facebookresearch/open-eqa/main/data/open-eqa-v0.json"
HM3D_FRAMES_URL="https://www.dropbox.com/scl/fi/t79gsjqlan8dneg7o63sw/open-eqa-hm3d-frames-v0.tgz?rlkey=1iuukwy2g3f5t06q4a3mxqobm&dl=1"
HM3D_FRAMES_MD5="286aa5d2fda99f4ed1567ae212998370"
HM3D_STATES_URL="https://www.dropbox.com/scl/fi/wg1uj1gvr4tkcz9aq3tzb/open-eqa-hm3d-states-v0.tgz?rlkey=i69chnpib8ui4cfabxa3iy9oj&dl=1"

OUTPUT_DIR="${1:-./source_data/openeqa}"
MANIFEST_OUT="${2:-./data/openeqa_manifest.jsonl}"
NUM_FRAMES="${3:-8}"
SCANNET_DIR="${4:-./source_data/scannet}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_DIR/frames"

echo "==> Downloading QA file -> ${OUTPUT_DIR}/open-eqa-v0.json"
if [[ -f "$OUTPUT_DIR/open-eqa-v0.json" ]]; then
  echo "  skip (exists)"
else
  curl -fL "$QA_URL" -o "$OUTPUT_DIR/open-eqa-v0.json"
fi

echo "==> Downloading HM3D RGB frames (~12GB)"
if [[ -d "$OUTPUT_DIR/frames/hm3d-v0" ]]; then
  echo "  skip (already extracted): frames/hm3d-v0"
else
  TGZ="$OUTPUT_DIR/open-eqa-hm3d-frames-v0.tgz"
  curl -fL "$HM3D_FRAMES_URL" -o "$TGZ"
  echo "  verifying md5..."
  actual_md5="$(md5sum "$TGZ" | cut -d' ' -f1)"
  if [[ "$actual_md5" != "$HM3D_FRAMES_MD5" ]]; then
    echo "  WARNING: md5 mismatch (expected $HM3D_FRAMES_MD5, got $actual_md5) -- continuing anyway" >&2
  fi
  echo "  extracting..."
  tar -xzf "$TGZ" -C "$OUTPUT_DIR/frames"
  rm -f "$TGZ"
fi

echo "==> Downloading HM3D agent-state pickles (pose, small)"
if find "$OUTPUT_DIR/frames/hm3d-v0" -maxdepth 2 -name '*.pkl' -print -quit 2>/dev/null | grep -q .; then
  echo "  skip (states already present)"
else
  TGZ="$OUTPUT_DIR/open-eqa-hm3d-states-v0.tgz"
  curl -fL "$HM3D_STATES_URL" -o "$TGZ"
  echo "  extracting..."
  tar -xzf "$TGZ" -C "$OUTPUT_DIR/frames"
  rm -f "$TGZ"
fi

echo "==> NOTE: ScanNet-sourced episodes (scannet-v0/*) are NOT downloaded here --"
echo "    reusing --scannet-dir (${SCANNET_DIR}) if it has posed_images/, else"
echo "    those rows are skipped by convert_openeqa_manifest.py."

echo "==> Building manifest (pose-FPS ${NUM_FRAMES}-frame sampling) -> ${MANIFEST_OUT}"
bash "$SCRIPT_DIR/convert_openeqa_manifest.sh" "$OUTPUT_DIR" "$MANIFEST_OUT" "$NUM_FRAMES" "$SCANNET_DIR"

echo "==> Done."
