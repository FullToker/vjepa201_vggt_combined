#!/bin/bash
#SBATCH --job-name=dl_3rscan
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_3rscan_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_3rscan_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"

# FILE_TYPE defaults to sequence.zip (RGB-D frames). Override e.g.:
#   FILE_TYPE=semseg.v2.json sbatch dataset/download_3rscan.sbatch
FILE_TYPE="${FILE_TYPE:-sequence.zip}"

echo "Start download at $(date)"

srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    bash "$ROOT/dataset/download_3rscan.sh" \
        "$ROOT/source_data/3rscan" \
        "$FILE_TYPE"

echo "All done at $(date)"
