#!/bin/bash
#SBATCH --job-name=dl_leo_anno
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_leo_anno_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_leo_anno_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"

srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    bash "$ROOT/dataset/download_leo_annotations.sh" \
        "$ROOT/source_data/leo_annotations"

echo "All done at $(date)"
