#!/bin/bash
#SBATCH --job-name=mk_mmsibench
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/mk_mmsibench_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/mk_mmsibench_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs" "$ROOT/data"

srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    python3 "$ROOT/dataset/make_mmsibench_manifest.py" \
        --raw-jsonl "$ROOT/data/mmsibench_raw.jsonl" \
        --out "$ROOT/data/mmsibench_manifest.jsonl"

echo "All done at $(date)"
