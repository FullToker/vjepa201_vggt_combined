#!/bin/bash
#SBATCH --job-name=profile_gvjepa
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/profile_gvjepa/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/profile_gvjepa/%j.err

# Dedicated torch.profiler run — NOT a correctness/OOM check (see
# overfit_sft_from_vljepa_ckpt_fusiongv.sbatch for that). Single purpose:
# produce a TensorBoard trace of train + VSI-Bench infer so we can see where
# GPU/CPU time actually goes before touching any CUDA-level optimization
# (torch.compile, SDPA/FlashAttention, TF32, activation checkpointing, ...).
#
# --profile is opt-in and off by default everywhere else in this codebase
# (see fusion_gv/profiling.py) — this script is the one place that always
# passes it, so runs kicked off here always carry the profiling overhead;
# nothing else in the repo is affected by this script existing.
#
# Uses --overfit (frozen batch, repeated) purely to get a fast, deterministic
# handful of steps — the loss/accuracy numbers from this run are meaningless,
# same caveat as the overfit sanity check. train.output_dir below is a
# SEPARATE dir from both the real run and the overfit sanity check, so this
# never touches either of their checkpoints.
#
# Override the config to profile the vjepa-only path instead:
#   CONFIG=fusion_gv/configs/overfit_from_vljepa_ckpt.yaml sbatch fusion_gv/trainbatch/profile_gvjepa.sbatch
#
# View the result (paths follow CONFIG's train.output_dir / vsi_inference.output_dir --
# for the default fusiongv overfit config that's):
#   tensorboard --logdir outputs/sft_from_vljepa_ckpt_fusiongv_overfit/profiler_trace
#   tensorboard --logdir outputs/inference/sft_from_vljepa_ckpt_fusiongv_overfit/vsibench/profiler_trace

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/profile_gvjepa"
CONFIG="${CONFIG:-fusion_gv/configs/overfit_from_vljepa_ckpt_fusiongv.yaml}"
PROFILE_STEPS="${PROFILE_STEPS:-10}"

mkdir -p "$LOG_DIR"

SIF="srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    --env MLFLOW_ALLOW_FILE_STORE=true \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

# 1 (wait) + 1 (warmup) + PROFILE_STEPS (active) + a few steps of slack, per
# fusion_gv/profiling.py's default schedule.
OVERFIT_STEPS=$(( PROFILE_STEPS + 20 ))

echo "[profile-train] start $(date) config=$CONFIG profile_steps=$PROFILE_STEPS"
$SIF accelerate launch --num_processes 1 \
    fusion_gv/train_gvjepa.py \
    --config "$CONFIG" \
    --overfit \
    --overfit-samples 32 \
    --overfit-steps "$OVERFIT_STEPS" \
    --profile \
    --profile-steps "$PROFILE_STEPS"
echo "[profile-train] done $(date)"

echo "[profile-infer-vsibench] start $(date)"
$SIF python3 fusion_gv/infer_vsibench.py \
    --config "$CONFIG" \
    --config-section vsi_inference \
    --profile \
    --profile-steps "$PROFILE_STEPS"
echo "[profile-infer-vsibench] done $(date)"
