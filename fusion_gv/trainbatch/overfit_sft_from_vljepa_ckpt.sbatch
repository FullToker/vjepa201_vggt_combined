#!/bin/bash
#SBATCH --job-name=overfit_sft_vljepa
#SBATCH --partition=geo
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/overfit_sft_vljepa_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/overfit_sft_vljepa_%j.err

# Sanity check before the real sft_from_vljepa_ckpt.sbatch run:
#   - freezes one batch (128 samples, matches train.batch_size in the yaml so
#     per-GPU tensor shapes/memory match the real run) and repeats it
#   - 200 steps is enough to see loss drop AND to know within ~minutes
#     whether batch_size=128 (global 256 across 2 GPUs) OOMs
# Not a throughput benchmark — _build_overfit_loader repeats the same fixed
# batch, so step/sec here does not predict the real run's dataloader-bound
# throughput.

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"

MONITOR_LOG="$ROOT/logs/gpu_monitor_${SLURM_JOB_ID}.log"

nvidia-smi dmon -s pucvmet -d 30 > "$MONITOR_LOG" 2>&1 &

# Host RAM alongside GPU stats in the same log — this is exactly the sanity
# check for the DataLoader-worker-OOM seen on the previous run.
( while true; do
    printf '[MEM %s] used=%s / total=%s\n' \
        "$(date +%T)" \
        "$(free -h | awk '/Mem:/{print $3}')" \
        "$(free -h | awk '/Mem:/{print $2}')" \
        >> "$MONITOR_LOG"
    sleep 30
done ) &

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
        --config fusion_gv/configs/sft_from_vljepa_ckpt.yaml \
        --overfit \
        --overfit-samples 128 \
        --overfit-steps 200
