#!/usr/bin/env bash
# Downloads one ScanNet scene's official cleaned mesh (_vh_clean_2.ply) via the
# vendored official downloader. Used as real ground-truth geometry for VGGT
# reconstruction-quality comparisons (see evals/vggt_frame_count_ablation.py) --
# NOT reconstructed from depth+pose, NOT a VGGT self-reconstruction.
#
# Requires agreeing to the ScanNet TOS interactively; piped here so it works
# non-interactively (matches --skip_existing so re-running is a no-op).
#
# Usage:
#   bash dataset/download_scannet_scene_mesh.sh <scene_id> [out_dir]
#   bash dataset/download_scannet_scene_mesh.sh scene0002_00 ./source_data/scannet_mesh
#
# Output: <out_dir>/scans/<scene_id>/<scene_id>_vh_clean_2.ply
set -euo pipefail

SCENE_ID="${1:?Usage: download_scannet_scene_mesh.sh <scene_id> [out_dir]}"
OUT_DIR="${2:-./source_data/scannet_mesh}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Downloading ScanNet mesh for ${SCENE_ID} -> ${OUT_DIR}"
printf '\n' | python3 "${SCRIPT_DIR}/vendor_scannet_download.py" \
  --id "${SCENE_ID}" \
  --type _vh_clean_2.ply \
  --skip_existing \
  -o "${OUT_DIR}"

MESH_PATH="${OUT_DIR}/scans/${SCENE_ID}/${SCENE_ID}_vh_clean_2.ply"
if [[ -f "${MESH_PATH}" ]]; then
  echo "Done: ${MESH_PATH}"
else
  echo "ERROR: expected mesh not found at ${MESH_PATH}" >&2
  exit 1
fi
