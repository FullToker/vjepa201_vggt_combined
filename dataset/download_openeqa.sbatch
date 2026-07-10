#!/bin/bash
#SBATCH --job-name=dl_openeqa
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_openeqa_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_openeqa_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
SIF="srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

mkdir -p "$ROOT/logs"

echo "Start download at $(date)"

$SIF bash "$ROOT/dataset/download_openeqa.sh" \
    "$ROOT/source_data/openeqa" \
    "$ROOT/data/openeqa_manifest.jsonl" \
    8 \
    "$ROOT/source_data/scannet"

echo "All done at $(date)"
