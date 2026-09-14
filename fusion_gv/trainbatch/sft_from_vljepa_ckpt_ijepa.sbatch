#!/bin/bash
#SBATCH --job-name=sft_from_vljepa_ckpt_ijepa
#SBATCH --partition=geo
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_ijepa/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_ijepa/%j.err

# 3 GPUs, single job: train, then VSI-Bench infer, then MMSI-Bench infer,
# all against fusion_gv/configs/sft_from_vljepa_ckpt_ijepa.yaml. Sibling of
# sft_from_vljepa_ckpt.sbatch (x_encoder_type=vjepa) -- only the X-encoder
# differs (ijepa vs vjepa), everything else (data, hparams, GPU count,
# max_steps) is identical so the two runs' final.pt/eval scores are directly
# comparable as a video-vs-image-pretraining ablation.
#
# Same --mem=200G rationale as the vjepa baseline (prior host-RAM OOM at
# MaxRSS=155GB) -- not re-derived for this run, carried over defensively.
#
# All logs live under logs/sft_from_vljepa_ckpt_ijepa/ instead of flat in
# logs/. The #SBATCH --output/--error paths above are opened by Slurm before
# this script body runs, so mkdir -p inside the script is too late for
# THOSE two -- create the folder once before first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_ijepa
#
# Prereqs (run once, before this job):
#   sbatch dataset/download_open_vljepa_ckpt.sbatch    # -> ckpts/open_vljepa/best.pt
#   sbatch dataset/test_forward_ijepa.sbatch            # -> ckpts/IN1K-vit.h.14-300e.pth.tar (also smoke-tests the ijepa path)
#   sbatch dataset/convert_spar_sftqa_uni8.sbatch       # -> data/spar_sftqa_sftvljepa_uni8.jsonl
#   dataset/make_vsibench_manifest.py / make_mmsibench_manifest.py already run # -> data/{vsibench,mmsibench}_manifest.jsonl
#
# Run the overfit check first (same config, K=1, small step count) before
# committing to the full 7-day run -- mirrors
# fusion_gv/trainbatch/overfit_sft_from_vljepa_ckpt.sbatch's role for the
# vjepa baseline. No dedicated overfit_*_ijepa.sbatch exists yet; adapt that
# one's --config to sft_from_vljepa_ckpt_ijepa.yaml if you want one.
#
# To re-run only the infer steps after training already finished:
#   RUN_TRAIN=0 sbatch fusion_gv/trainbatch/sft_from_vljepa_ckpt_ijepa.sbatch
# Requires train.output_dir's final.pt (or step_*.pt) to already exist.

set -euo pipefail
RUN_TRAIN=${RUN_TRAIN:-1}

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_from_vljepa_ckpt_ijepa"

mkdir -p "$LOG_DIR"

MONITOR_LOG="$LOG_DIR/gpu_monitor_${SLURM_JOB_ID}.log"

nvidia-smi dmon -s pucvmet -d 30 > "$MONITOR_LOG" 2>&1 &

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

if [ "$RUN_TRAIN" = "1" ]; then
    echo "[train] start $(date)"
    $SIF accelerate launch --multi_gpu --num_processes 3 \
        fusion_gv/train_gvjepa.py \
        --config fusion_gv/configs/sft_from_vljepa_ckpt_ijepa.yaml
    echo "[train] done $(date)"
else
    echo "[train] skipped (RUN_TRAIN=0), reusing existing checkpoint"
fi

echo "[infer-vsibench] start $(date)"
$SIF python3 fusion_gv/infer_vsibench.py \
    --config fusion_gv/configs/sft_from_vljepa_ckpt_ijepa.yaml \
    --config-section vsi_inference
echo "[infer-vsibench] done $(date)"

echo "[infer-mmsibench] start $(date)"
$SIF python3 fusion_gv/infer_mmsibench.py \
    --config fusion_gv/configs/sft_from_vljepa_ckpt_ijepa.yaml \
    --config-section mmsi_inference
echo "[infer-mmsibench] done $(date)"
