#!/bin/bash
#SBATCH --job-name=overfit_sft_vljepa_fusiongv
#SBATCH --partition=geo
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv/overfit_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv/overfit_%j.err

# Sanity check before the real sft_from_vljepa_ckpt_fusiongv.sbatch run:
# train(--overfit) -> VSI-Bench infer -> MMSI-Bench infer, all against the
# SEPARATE fusion_gv/configs/overfit_from_vljepa_ckpt_fusiongv.yaml (own
# output_dir, own infer output_dirs -- never touches the real run's paths,
# see that yaml's header).
#   - freezes one batch (32 samples, matches train.batch_size in that yaml so
#     per-GPU tensor shapes/memory match the real run -- _build_overfit_loader
#     hands every rank the same full fixed batch, unsharded, so this is a
#     per-GPU number, not multiplied by the 3 GPUs) and repeats it
#   - 200 steps is enough to see loss drop AND to know within ~minutes
#     whether batch_size=32 (fusion_gv: VGGT+JEPA both loaded per sample) OOMs
#   - the infer steps then confirm the whole train->checkpoint->infer chain
#     runs end-to-end without crashing (accuracy numbers from an overfit
#     checkpoint are meaningless, this is a plumbing check, not an eval)
# Not a throughput benchmark — _build_overfit_loader repeats the same fixed
# batch, so step/sec here does not predict the real run's dataloader-bound
# throughput.
#
# Shares logs/sft_from_vljepa_ckpt_fusiongv/ with the real run (overfit_
# prefix keeps filenames apart) -- the #SBATCH --output/--error paths above
# are opened by Slurm before this script body runs, so mkdir -p inside the
# script is too late for those two -- create the folder once before first
# submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_from_vljepa_ckpt_fusiongv"
CONFIG="fusion_gv/configs/overfit_from_vljepa_ckpt_fusiongv.yaml"

mkdir -p "$LOG_DIR"

MONITOR_LOG="$LOG_DIR/gpu_monitor_overfit_${SLURM_JOB_ID}.log"

nvidia-smi dmon -s pucvmet -d 30 > "$MONITOR_LOG" 2>&1 &

# Host RAM alongside GPU stats in the same log — fusion_gv loads both VGGT
# and JEPA per sample, more host-RAM-hungry per DataLoader worker than the
# vjepa-only run that OOM'd on host RAM before.
( while true; do
    printf '[MEM %s] used=%s / total=%s\n' \
        "$(date +%T)" \
        "$(free -h | awk '/Mem:/{print $3}')" \
        "$(free -h | awk '/Mem:/{print $2}')" \
        >> "$MONITOR_LOG"
    sleep 30
done ) &

SIF="srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    --env MLFLOW_ALLOW_FILE_STORE=true \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

echo "[overfit-train] start $(date)"
$SIF accelerate launch --multi_gpu --num_processes 3 \
    fusion_gv/train_gvjepa.py \
    --config "$CONFIG" \
    --overfit \
    --overfit-samples 32 \
    --overfit-steps 200
echo "[overfit-train] done $(date)"

# Infer steps are single-GPU (infer_vsibench.py/infer_mmsibench.py don't use
# accelerate) -- fine to run on 1 of the 3 allocated GPUs after training ends.
echo "[overfit-infer-vsibench] start $(date)"
$SIF python3 fusion_gv/infer_vsibench.py \
    --config "$CONFIG" \
    --config-section vsi_inference
echo "[overfit-infer-vsibench] done $(date)"

echo "[overfit-infer-mmsibench] start $(date)"
$SIF python3 fusion_gv/infer_mmsibench.py \
    --config "$CONFIG" \
    --config-section mmsi_inference
echo "[overfit-infer-mmsibench] done $(date)"
