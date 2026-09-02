#!/bin/bash
#SBATCH --job-name=overfit_sft_vljepa_fusiongv_3gpu
#SBATCH --partition=geo
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv/overfit3gpu_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv/overfit3gpu_%j.err

# Quick 3-GPU DDP memory check: K=1 (default, not set in the yaml),
# batch_size=32 -- same config overfit_sft_from_vljepa_ckpt_fusiongv.sbatch
# uses, but that one is single-GPU (a plumbing check only, doesn't need to
# match GPU count). This one specifically tests real 3-GPU DDP overhead at
# K=1 -- the same overhead (output-tensor clone in _DDPSink, gradient
# bucketing) that pushed K=128/batch=32 over the edge on the real SFT run
# (see chat history: fixed there by batch_size 32->16). Single-GPU profiling
# measured K=1 at only 40.7GB (well under the 79GB ceiling), so this is
# expected to pass easily, but confirming beats assuming, same as K=128 was
# assumed-fine from single-GPU numbers and then didn't fit in real 3-GPU DDP.
#
# Train only, no infer -- this is a memory/plumbing check, not an eval.
# --output-dir isolates this run from the 1-GPU overfit's checkpoint (same
# yaml, own output_dir).
#
# Log dir must exist before first submission (already true if
# overfit_sft_from_vljepa_ckpt_fusiongv.sbatch ran before; otherwise:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_from_vljepa_ckpt_fusiongv"
CONFIG="fusion_gv/configs/overfit_from_vljepa_ckpt_fusiongv.yaml"
OUTPUT_DIR="./outputs/sft_from_vljepa_ckpt_fusiongv_overfit_3gpu"

mkdir -p "$LOG_DIR"

MONITOR_LOG="$LOG_DIR/gpu_monitor_overfit3gpu_${SLURM_JOB_ID}.log"
nvidia-smi dmon -s pucvmet -d 15 > "$MONITOR_LOG" 2>&1 &

SIF="srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    --env MLFLOW_ALLOW_FILE_STORE=true \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

echo "[overfit-train-3gpu] start $(date)"
$SIF accelerate launch --multi_gpu --num_processes 3 \
    fusion_gv/train_gvjepa.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --overfit \
    --overfit-samples 32 \
    --overfit-steps 50
echo "[overfit-train-3gpu] done $(date)"
