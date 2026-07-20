#!/bin/bash
#SBATCH --job-name=dl_scannet_mesh
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_scannet_mesh_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_scannet_mesh_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
SIF="srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

mkdir -p "$ROOT/logs"

# Which scene to fetch -- override at submit time:
#   sbatch --export=SCENE_ID=scene0002_00 dataset/download_scannet_scene_mesh.sbatch
SCENE_ID="${SCENE_ID:-scene0011_00}"
OUT_DIR="${OUT_DIR:-$ROOT/source_data/scannet_mesh}"

echo "Start ScanNet mesh download for ${SCENE_ID} at $(date)"

$SIF bash "$ROOT/dataset/download_scannet_scene_mesh.sh" \
    "$SCENE_ID" \
    "$OUT_DIR"

echo "All done at $(date)"
