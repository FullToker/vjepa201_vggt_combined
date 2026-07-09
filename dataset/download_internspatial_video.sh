#!/usr/bin/env bash
# Downloads only the multi-view ("video_*.parquet") shard of InternSpatial.
# Each row bundles 15+ frames + conversations (multi-view spatial QA).
# The single-view "data_*.parquet" shard (~412GB) is skipped — not needed here.
#
# Usage:
#   HF_TOKEN=hf_xxx bash dataset/download_internspatial_video.sh [OUTPUT_DIR]
#   OR: hf auth login first, then run without HF_TOKEN.
#
# Dataset page: https://huggingface.co/datasets/Yeshenglong/InternSpatial
set -euo pipefail
export HF_HUB_DOWNLOAD_TIMEOUT=60 

REPO_ID="Yeshenglong/InternSpatial"
OUTPUT_DIR="${1:-./source_data/internspatial}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf'. Install: pip install huggingface_hub[cli]" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARG=(--token "$HF_TOKEN")
fi

echo "Downloading InternSpatial video_*.parquet (multi-view) to: ${OUTPUT_DIR}"

MAX_RETRY=30
for i in $(seq 1 $MAX_RETRY); do
  echo "=== Attempt $i at $(date) ==="
  if hf download "$REPO_ID" \
    --repo-type dataset \
    --include "video_*.parquet" \
    --local-dir "$OUTPUT_DIR" \
     --max-workers 2 \
    "${TOKEN_ARG[@]}"; then
    echo "Download completed successfully at $(date)"
    exit 0
  else
    echo "Attempt $i failed at $(date), retrying in 30s..."
    sleep 30
  fi
done

echo "All $MAX_RETRY attempts failed." >&2
exit 1