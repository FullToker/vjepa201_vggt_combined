#!/usr/bin/env bash
# Downloads EmbodiedScan annotation PKLs from HuggingFace.
# Images (ScanNet/HM3D/MP3D) require separate data-use agreements — see README.
#
# Usage:
#   bash dataset/download_embodiedscan.sh [OUTPUT_DIR]
#
# Verify filenames at:
#   https://huggingface.co/datasets/OpenRobotLab/EmbodiedScan/tree/main
set -euo pipefail

REPO_ID="OpenRobotLab/EmbodiedScan"
OUTPUT_DIR="${1:-./source_data/embodiedscan}"

if ! command -v wget >/dev/null 2>&1; then
  echo "Missing 'wget'. Install: sudo apt install wget" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

BASE_URL="https://huggingface.co/datasets/${REPO_ID}/resolve/main"

FILES=(
  "embodiedscan_infos_train_full.pkl"
  "embodiedscan_infos_val_full.pkl"
)

echo "Downloading EmbodiedScan PKLs to: ${OUTPUT_DIR}"
for f in "${FILES[@]}"; do
  wget -c "${BASE_URL}/${f}" -P "$OUTPUT_DIR"
done

echo "Done. Images (ScanNet/HM3D/MP3D) need separate download per their data agreements."
