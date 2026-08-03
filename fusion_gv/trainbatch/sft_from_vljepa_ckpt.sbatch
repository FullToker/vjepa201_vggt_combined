#!/bin/bash
#SBATCH --job-name=sft_from_vljepa_ckpt
#SBATCH --partition=geo
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_%j.err

# Full SFT run: 3 epochs over spar_sftqa_sftvljepa_uni8.jsonl (1,771,283 rows),
# global batch = batch_size(128) * 2 GPUs = 256 -> max_steps=20757 (see yaml).
#
# Prereqs (run once, before this job):
#   sbatch dataset/download_open_vljepa_ckpt.sbatch   # -> ckpts/open_vljepa/best.pt
#   sbatch dataset/convert_spar_sftqa_uni8.sbatch      # -> data/spar_sftqa_sftvljepa_uni8.jsonl (done)
#   sbatch fusion_gv/trainbatch/overfit_sft_from_vljepa_ckpt.sbatch   # OOM/correctness check first

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
        --config fusion_gv/configs/sft_from_vljepa_ckpt.yaml
