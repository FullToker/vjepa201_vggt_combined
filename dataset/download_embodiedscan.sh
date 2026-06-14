#!/usr/bin/env bash
# Downloads EmbodiedScan annotation PKLs from HuggingFace.
# Images (ScanNet/HM3D/MP3D) require separate data-use agreements — see README.
#
# Usage:
#   HF_TOKEN=hf_xxx bash dataset/download_embodiedscan.sh [OUTPUT_DIR]
#   OR: huggingface-cli login first, then run without HF_TOKEN.
#
# Dataset page: https://huggingface.co/datasets/OpenRobotLab/EmbodiedScan
set -euo pipefail

REPO_ID="cjfcsjt/embodiedscan"
OUTPUT_DIR="${1:-./source_data/embodiedscan}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf'. Install: pip install huggingface_hub[cli]" >&2
  exit 1
fi


mkdir -p "$OUTPUT_DIR"

FILES=(
  "embodiedscan.zip"
)

# Pass token via env if set; otherwise huggingface-cli uses cached login.
TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARG=(--token "$HF_TOKEN")
fi

echo "Downloading EmbodiedScan PKLs to: ${OUTPUT_DIR}"
for f in "${FILES[@]}"; do
  hf download "$REPO_ID" "$f" \
    --repo-type dataset \
    --local-dir "$OUTPUT_DIR" \
    "${TOKEN_ARG[@]}"
done

echo "Extracting embodiedscan.zip..."
python3 -m zipfile -e "${OUTPUT_DIR}/embodiedscan.zip" "${OUTPUT_DIR}"
rm "${OUTPUT_DIR}/embodiedscan.zip"
echo "Done. Extracted to: ${OUTPUT_DIR}"
