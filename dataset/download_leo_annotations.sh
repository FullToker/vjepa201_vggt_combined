#!/usr/bin/env bash
# Downloads LEO's released text annotations (alignment + instruction stages)
# -- object/scene-referring descriptions tied to ScanNet scan_id/target_id
# (ScanRefer, ReferIt3D Nr3D/Sr3D+) plus 3RScan scene captions and Objaverse
# object captions. Only annotations.zip is pulled (~62.5MB, text only) --
# NOT LEO's point clouds/checkpoints (pcd_with_global_alignment.zip,
# 3RScan-ours-align.zip, *.pth, etc: those assume you don't already have
# ScanNet, which you do).
#
# bbox coordinates are NOT included here -- resolve target_id against your
# own ScanNet instance-segmentation annotations
# (sceneXXXX_XX.aggregation.json / .segs.json) to get object boxes; that's a
# separate conversion step, not part of this download.
#
# Usage:
#   HF_TOKEN=hf_xxx bash dataset/download_leo_annotations.sh [SOURCE_DIR]
#   OR: hf auth login first, then run without HF_TOKEN.
#
# Dataset page: https://huggingface.co/datasets/huangjy-pku/LEO_data
# Paper: LEO, ICML 2024, https://arxiv.org/abs/2311.12871
set -euo pipefail

REPO_ID="huangjy-pku/LEO_data"
SOURCE_DIR="${1:-./source_data/leo_annotations}"

mkdir -p "$SOURCE_DIR"

TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARG=(--token "$HF_TOKEN")
fi

if [[ -d "$SOURCE_DIR/annotations" ]]; then
  echo "==> Already extracted, skipping: $SOURCE_DIR/annotations"
  exit 0
fi

echo "==> Downloading annotations.zip to: ${SOURCE_DIR}"
hf download "$REPO_ID" annotations.zip \
  --repo-type dataset \
  --local-dir "$SOURCE_DIR" \
  "${TOKEN_ARG[@]}"

echo "==> Extracting annotations.zip"
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
  "$SOURCE_DIR/annotations.zip" "$SOURCE_DIR"
rm -f "$SOURCE_DIR/annotations.zip"

echo "==> Done. Layout:"
echo "    $SOURCE_DIR/annotations/alignment/{obj_caption,obj_scene_caption,scene_caption}"
echo "    $SOURCE_DIR/annotations/instruction/{scan2cap,scanqa,sqa3d,3rscanqa,dialogue,planning}"
