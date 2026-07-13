#!/bin/bash
#SBATCH --job-name=dl_scannet_posed
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_scannet_posed_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_scannet_posed_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"
source "$ROOT/.env/hf.env"

echo "Start download at $(date)"

srun --export=ALL singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    bash "$ROOT/dataset/download_scannet_posed.sh" \
        "$ROOT/source_data/scannet"

echo "All done at $(date)"
