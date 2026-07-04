#!/bin/bash
#SBATCH --job-name=sft_grounding_mixed
#SBATCH --partition=geo
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=122:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_grounding_mixed_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_grounding_mixed_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"

nvidia-smi dmon -s pucvmet -d 30 > "$ROOT/logs/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &

srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    --env MLFLOW_ALLOW_FILE_STORE=true \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    accelerate launch --multi_gpu --num_processes 2 \
        fusion_gv/train_gvjepa.py \
        --config fusion_gv/configs/sft_grounding_mixed.yaml
