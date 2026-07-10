#!/bin/bash
#SBATCH --job-name=extract_vsibench_frames
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/extract_vsibench_frames_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/extract_vsibench_frames_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
SIF="srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

mkdir -p "$ROOT/logs"

echo "Start frame extraction at $(date)"

$SIF bash "$ROOT/dataset/extract_vsibench_frames.sh" \
    "$ROOT/source_data/vsibench" \
    "$ROOT/data/vsibench_manifest.jsonl" \
    4

echo "All done at $(date)"
