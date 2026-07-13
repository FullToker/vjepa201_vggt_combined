#!/bin/bash
#SBATCH --job-name=dl_scannet_2dinst
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_scannet_2dinst_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_scannet_2dinst_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"

# LIMIT=0 (default) downloads all train/val scans. Set LIMIT=3 for a smoke
# test first, e.g.: LIMIT=3 sbatch dataset/download_scannet_2d_instance.sbatch
LIMIT="${LIMIT:-0}"

echo "Start download at $(date)"

srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    bash "$ROOT/dataset/download_scannet_2d_instance.sh" \
        "$ROOT/source_data/scannet" \
        "$LIMIT"

echo "All done at $(date)"
