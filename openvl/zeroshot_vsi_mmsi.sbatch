#!/bin/bash
#SBATCH --job-name=openvl_zeroshot
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/openvl_zeroshot/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/openvl_zeroshot/%j.err

# Zero-shot open-vljepa baseline (raw checkpoint, no SFT) on VSI-Bench +
# MMSI-Bench. Single GPU, inference-only. See openvl/infer_bench.py header
# for what this measures and why (baseline to compare against the
# warm-started+SFT'd fusion_gv runs).
#
# Assumes ../open-vljepa is checked out as a sibling of this project's root
# (default in openvl/infer_bench.py) -- pass --openvljepa-repo below if your
# cluster layout differs.
#
# Create the log folder once before first submission (Slurm opens
# --output/--error before this script body runs):
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/openvl_zeroshot
#
# Prereq: sbatch dataset/download_open_vljepa_ckpt.sbatch   # -> ckpts/open_vljepa/best.pt

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/openvl_zeroshot"

mkdir -p "$LOG_DIR"

SIF="srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

echo "[zeroshot-vsibench] start $(date)"
$SIF python3 openvl/infer_bench.py \
    --config openvl/configs/zeroshot_infer.yaml \
    --config-section vsi_inference
echo "[zeroshot-vsibench] done $(date)"

echo "[zeroshot-mmsibench] start $(date)"
$SIF python3 openvl/infer_bench.py \
    --config openvl/configs/zeroshot_infer.yaml \
    --config-section mmsi_inference
echo "[zeroshot-mmsibench] done $(date)"
