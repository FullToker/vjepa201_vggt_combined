#!/bin/bash
#SBATCH --job-name=infer_vsibench
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/infer_vsibench_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/infer_vsibench_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"

srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    python3 fusion_gv/infer_vsibench.py \
        --config fusion_gv/configs/infer_vsibench_mixedsft_8select.yaml \
        --config-section inference
