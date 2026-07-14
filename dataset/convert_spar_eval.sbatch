#!/bin/bash
#SBATCH --job-name=convert_spar_eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/convert_spar_eval_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/convert_spar_eval_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
SIF="srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

mkdir -p "$ROOT/logs"

echo "Start eval regen at $(date)"

$SIF python3 "$ROOT/dataset/convert_spar_sftqa.py" \
    --spar-root "$ROOT/source_data/spar" \
    --output-dir "$ROOT/data" \
    --eval-output spar_sftqa_eval.jsonl \
    --only eval \
    --include-metadata

echo "All done at $(date)"
