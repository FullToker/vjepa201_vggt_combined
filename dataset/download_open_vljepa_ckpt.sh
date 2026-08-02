#!/usr/bin/env bash
# Downloads the open-vljepa trained checkpoint (best.pt) from HuggingFace.
# Used as init_vljepa_ckpt in fusion_gv/configs/sft_from_vljepa_ckpt.yaml to
# warm-start predictor + y_encoder before fusion_gv SFT — see
# fusion_gv/load_vljepa_init.py for what transfers.
#
# Usage:
#   HF_TOKEN=hf_xxx bash dataset/download_open_vljepa_ckpt.sh [OUTPUT_DIR]
#   OR: huggingface-cli login first, then run without HF_TOKEN.
#
# Model page: https://huggingface.co/cun-bjy/open-vljepa
set -euo pipefail

REPO_ID="cun-bjy/open-vljepa"
OUTPUT_DIR="${1:-./ckpts/open_vljepa}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf'. Install: pip install huggingface_hub[cli]" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARG=(--token "$HF_TOKEN")
fi

echo "Downloading open-vljepa best.pt to: ${OUTPUT_DIR}"
hf download "$REPO_ID" best.pt \
  --repo-type model \
  --local-dir "$OUTPUT_DIR" \
  "${TOKEN_ARG[@]}"

echo "Done. Checkpoint at: ${OUTPUT_DIR}/best.pt"
