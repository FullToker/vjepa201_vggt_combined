#!/usr/bin/env bash
# Downloads VSI-Bench (annotations + video zips) and builds a multiple-choice
# inference manifest (candidate-comparison eval, no y-decoder needed).
#
# Usage:
#   HF_TOKEN=hf_xxx bash dataset/download_vsibench.sh [SOURCE_DIR] [MANIFEST_OUT]
#   OR: hf auth login first, then run without HF_TOKEN.
#
# Dataset page: https://huggingface.co/datasets/nyu-visionx/VSI-Bench
set -euo pipefail

REPO_ID="nyu-visionx/VSI-Bench"
SOURCE_DIR="${1:-./source_data/vsibench}"
MANIFEST_OUT="${2:-./data/vsibench_manifest.jsonl}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf'. Install: pip install huggingface_hub[cli]" >&2
  exit 1
fi

mkdir -p "$SOURCE_DIR"

TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARG=(--token "$HF_TOKEN")
fi

ANNOTATION_FILES=(test.jsonl test_debiased.parquet test_pruned.parquet pruned_ids.txt)
VIDEO_ZIPS=(arkitscenes.zip scannet.zip scannetpp.zip)

echo "==> Downloading annotations to: ${SOURCE_DIR}"
for fname in "${ANNOTATION_FILES[@]}"; do
  if [[ -f "$SOURCE_DIR/$fname" ]]; then
    echo "  skip (exists): $fname"
    continue
  fi
  hf download "$REPO_ID" "$fname" \
    --repo-type dataset \
    --local-dir "$SOURCE_DIR" \
    "${TOKEN_ARG[@]}"
  echo "  downloaded: $fname"
done

echo "==> Downloading + extracting video zips (large)"
for fname in "${VIDEO_ZIPS[@]}"; do
  extracted_dir="$SOURCE_DIR/${fname%.zip}"
  if [[ -d "$extracted_dir" ]]; then
    echo "  skip (already extracted): $fname"
    continue
  fi
  if [[ ! -f "$SOURCE_DIR/$fname" ]]; then
    hf download "$REPO_ID" "$fname" \
      --repo-type dataset \
      --local-dir "$SOURCE_DIR" \
      "${TOKEN_ARG[@]}"
    echo "  downloaded: $fname"
  fi
  echo "  extracting: $fname ..."
  unzip -q -o "$SOURCE_DIR/$fname" -d "$SOURCE_DIR"
  rm -f "$SOURCE_DIR/$fname"
  echo "  extracted and removed zip: $fname"
done

echo "==> Building multiple-choice manifest -> ${MANIFEST_OUT}"
python3 "$SCRIPT_DIR/make_vsibench_manifest.py" \
  --vsi-dir "$SOURCE_DIR" \
  --out "$MANIFEST_OUT" \
  --mode mc

echo "==> Done."
