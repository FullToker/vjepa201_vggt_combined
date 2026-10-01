#!/bin/bash
#SBATCH --job-name=overfit_pretrain_sft
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft/overfit_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft/overfit_%j.err

# Single-GPU plumbing check for the whole experiment, before the two 3-GPU runs:
#   1. stage 1 --overfit (128 frozen samples, 200 steps)        -> outputs/pretrain_vjepa_sceneverse_overfit/final.pt
#   2. stage 2 --overfit, warm-started from that final.pt        -> outputs/sft_pretrain_vjepa_sceneverse_overfit/final.pt
#      (this is what exercises init_gvjepa_ckpt and its strict weight load)
#   3. VSI-Bench and MMSI-Bench inference on the stage-2 checkpoint
# Per-GPU batch is 128 in both stages, the same as one rank of the real runs, so this also
# answers whether it fits in memory. Accuracy from an overfit checkpoint means nothing;
# what to look for is loss dropping in both stages and the inference steps finishing.
#
# Logs for the whole experiment: logs/pretrain_vjepa_sceneverse_sft/ (create once before
# the first submission):
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft
#
# Prereqs: ckpts/open_vljepa/best.pt, data/sceneverse_scene_pretrain_train.jsonl,
#          data/spar_sftqa_sftvljepa_uni8.jsonl, data/{vsibench,mmsibench}_manifest.jsonl
#
# Skip a finished stage when iterating: RUN_STAGE1=0 / RUN_STAGE2=0 (the later stages need
# the earlier stage's final.pt to exist).

set -euo pipefail
RUN_STAGE1=${RUN_STAGE1:-1}
RUN_STAGE2=${RUN_STAGE2:-1}

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/pretrain_vjepa_sceneverse_sft"
PRETRAIN_CFG="fusion_gv/configs/overfit_pretrain_vjepa_sceneverse.yaml"
SFT_CFG="fusion_gv/configs/overfit_sft_pretrain_vjepa_sceneverse.yaml"

cd "$ROOT"
mkdir -p "$LOG_DIR"

MONITOR_LOG="$LOG_DIR/overfit_gpu_monitor_${SLURM_JOB_ID}.log"

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

if [ "$RUN_STAGE1" = "1" ]; then
    echo "[overfit-stage1] start $(date)"
    $SIF accelerate launch --num_processes 1 \
        fusion_gv/train_gvjepa.py \
        --config "$PRETRAIN_CFG" \
        --overfit --overfit-samples 64 --overfit-steps 200
    echo "[overfit-stage1] done $(date)"
else
    echo "[overfit-stage1] skipped (RUN_STAGE1=0)"
fi

if [ "$RUN_STAGE2" = "1" ]; then
    echo "[overfit-stage2] start $(date)"
    $SIF accelerate launch --num_processes 1 \
        fusion_gv/train_gvjepa.py \
        --config "$SFT_CFG" \
        --overfit --overfit-samples 64 --overfit-steps 200
    echo "[overfit-stage2] done $(date)"
else
    echo "[overfit-stage2] skipped (RUN_STAGE2=0)"
fi

echo "[overfit-infer-vsibench] start $(date)"
$SIF python3 fusion_gv/infer_vsibench.py \
    --config "$SFT_CFG" \
    --config-section vsi_inference
echo "[overfit-infer-vsibench] done $(date)"

echo "[overfit-infer-mmsibench] start $(date)"
$SIF python3 fusion_gv/infer_mmsibench.py \
    --config "$SFT_CFG" \
    --config-section mmsi_inference
echo "[overfit-infer-mmsibench] done $(date)"
