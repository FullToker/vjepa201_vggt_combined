#!/usr/bin/env bash
# Downloads ScanNet posed images (EmbodiedScan subset) from HuggingFace.
# Source: GUESSGUO/annotated_posed_images_scannet (115 GB, 11 parts)
# Each part is ~10.7 GB. All parts form a single split tar.gz archive.
#
# Usage:
#   HF_TOKEN=hf_xxx bash dataset/download_scannet_posed.sh [OUTPUT_DIR]
#   OR: hf login first, then run without HF_TOKEN.
set -euo pipefail

REPO_ID="GUESSGUO/annotated_posed_images_scannet"
OUTPUT_DIR="${1:-./source_data/scannet}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf'. Install: pip install huggingface_hub[cli]" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARG=(--token "$HF_TOKEN")
fi

PARTS=(
  "annotated.tar.gz.part00"
  "annotated.tar.gz.part01"
  "annotated.tar.gz.part02"
  "annotated.tar.gz.part03"
  "annotated.tar.gz.part04"
  "annotated.tar.gz.part05"
  "annotated.tar.gz.part06"
  "annotated.tar.gz.part07"
  "annotated.tar.gz.part08"
  "annotated.tar.gz.part09"
  "annotated.tar.gz.part10"
)

echo "Downloading ScanNet posed images to: ${OUTPUT_DIR}"
for part in "${PARTS[@]}"; do
  echo "  -> ${part}"
  hf download "$REPO_ID" "$part" \
    --repo-type dataset \
    --local-dir "$OUTPUT_DIR" \
    "${TOKEN_ARG[@]}"
done

echo "All parts downloaded. Extracting..."
cat "${OUTPUT_DIR}"/annotated.tar.gz.part* | tar -xzf - -C "$OUTPUT_DIR"

echo "Removing part files..."
rm "${OUTPUT_DIR}"/annotated.tar.gz.part*

echo "Done. Extracted to: ${OUTPUT_DIR}/posed_images/"
