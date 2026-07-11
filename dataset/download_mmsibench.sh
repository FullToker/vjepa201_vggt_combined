#!/usr/bin/env bash
# Downloads MMSI-Bench (parquet, images embedded) and explodes it into
# jpg files + a raw jsonl manifest (id, images[], question, answer,
# question_type, difficulty). Images-per-row varies (2-10); no fixed-frame
# bucketing or MC-candidate parsing here -- that's a separate downstream step.
#
# Usage:
#   HF_TOKEN=hf_xxx bash dataset/download_mmsibench.sh [SOURCE_DIR] [MANIFEST_OUT]
#   OR: hf auth login first, then run without HF_TOKEN.
#
# Dataset page: https://huggingface.co/datasets/RunsenXu/MMSI-Bench
set -euo pipefail

REPO_ID="RunsenXu/MMSI-Bench"
SOURCE_DIR="${1:-./source_data/mmsibench}"
MANIFEST_OUT="${2:-./data/mmsibench_raw.jsonl}"
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

echo "==> Downloading parquet to: ${SOURCE_DIR}"
hf download "$REPO_ID" \
  --repo-type dataset \
  --include "*.parquet" \
  --local-dir "$SOURCE_DIR" \
  "${TOKEN_ARG[@]}"

echo "==> Exploding images + writing raw manifest -> ${MANIFEST_OUT}"
python3 "$SCRIPT_DIR/extract_mmsibench_images.py" \
  --src-dir "$SOURCE_DIR" \
  --out-jsonl "$MANIFEST_OUT"

echo "==> Done."
