#!/bin/bash
#SBATCH --job-name=overfit_grounding
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/overfit_grounding_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/overfit_grounding_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"

srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    python fusion_gv/train_gvjepa.py \
        --config fusion_gv/configs/overfit_grounding.yaml \
        --overfit \
        --overfit-samples 32 \
        --overfit-steps 500
