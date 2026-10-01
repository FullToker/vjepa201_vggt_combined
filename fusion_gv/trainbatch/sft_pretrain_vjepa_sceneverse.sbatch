#!/bin/bash
#SBATCH --job-name=sft_pretrain_vjepa_sceneverse
#SBATCH --partition=geo
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft/sft_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft/sft_%j.err

# Stage 2: SPAR SFT warm-started from the stage-1 checkpoint, then VSI-Bench and MMSI-Bench
# inference. 2 GPUs (was 3), V-JEPA only. Config: fusion_gv/configs/sft_pretrain_vjepa_sceneverse.yaml,
# which matches the sft_from_vljepa_ckpt.yaml baseline in everything but the init weights,
# so the scores compare directly (baseline: VSI 29.03%, MMSI 24.90%).
#
# Reads ./outputs/pretrain_vjepa_sceneverse/final.pt. To start from an earlier stage-1
# checkpoint (e.g. if its val loss turned up), edit init_gvjepa_ckpt in the yaml.
#
# Logs for the whole experiment: logs/pretrain_vjepa_sceneverse_sft/ (create once before
# the first submission):
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft
#
# Prereqs:
#   sbatch dataset/convert_spar_sftqa_uni8.sbatch                       # -> data/spar_sftqa_sftvljepa_uni8.jsonl
#   sbatch fusion_gv/trainbatch/pretrain_vjepa_sceneverse.sbatch        # -> outputs/pretrain_vjepa_sceneverse/final.pt
#   (data/{vsibench,mmsibench}_manifest.jsonl already built for the baseline)
#
# To re-run only the inference after training finished:
#   RUN_TRAIN=0 sbatch fusion_gv/trainbatch/sft_pretrain_vjepa_sceneverse.sbatch

set -euo pipefail
RUN_TRAIN=${RUN_TRAIN:-1}

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/pretrain_vjepa_sceneverse_sft"
CONFIG="fusion_gv/configs/sft_pretrain_vjepa_sceneverse.yaml"

cd "$ROOT"
mkdir -p "$LOG_DIR"

if [ "$RUN_TRAIN" = "1" ] && [ ! -f "$ROOT/outputs/pretrain_vjepa_sceneverse/final.pt" ]; then
    echo "missing stage-1 checkpoint: $ROOT/outputs/pretrain_vjepa_sceneverse/final.pt" >&2
    echo "run fusion_gv/trainbatch/pretrain_vjepa_sceneverse.sbatch first" >&2
    exit 1
fi

MONITOR_LOG="$LOG_DIR/sft_gpu_monitor_${SLURM_JOB_ID}.log"

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
    echo "[sft] start $(date)"
    $SIF accelerate launch --multi_gpu --num_processes 2 \
        fusion_gv/train_gvjepa.py \
        --config "$CONFIG"
    echo "[sft] done $(date)"
else
    echo "[sft] skipped (RUN_TRAIN=0), reusing existing checkpoint"
fi

echo "[infer-vsibench] start $(date)"
$SIF python3 fusion_gv/infer_vsibench.py \
    --config "$CONFIG" \
    --config-section vsi_inference
echo "[infer-vsibench] done $(date)"

echo "[infer-mmsibench] start $(date)"
$SIF python3 fusion_gv/infer_mmsibench.py \
    --config "$CONFIG" \
    --config-section mmsi_inference
echo "[infer-mmsibench] done $(date)"
