#!/bin/bash
#SBATCH --job-name=dl_internspatial_video
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_internspatial_video_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_internspatial_video_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
SIF="srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

mkdir -p "$ROOT/logs"

echo "Start download at $(date)"

$SIF bash "$ROOT/dataset/download_internspatial_video.sh" "$ROOT/source_data/internspatial"

echo "All done at $(date)"
