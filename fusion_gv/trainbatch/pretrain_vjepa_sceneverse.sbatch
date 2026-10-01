#!/bin/bash
#SBATCH --job-name=pretrain_vjepa_sceneverse
#SBATCH --partition=geo
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft/pretrain_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft/pretrain_%j.err

# Stage 1: scene-level language pretraining, 2 GPUs (was 3), V-JEPA only (no VGGT), with the
# frame-order shuffle on. Config: fusion_gv/configs/pretrain_vjepa_sceneverse.yaml
# (~600 steps; val loss every 100 steps on held-out spaces, printed in the log/MLflow).
# Writes ./outputs/pretrain_vjepa_sceneverse/{step_0000300.pt,final.pt}, which stage 2
# (sft_pretrain_vjepa_sceneverse.sbatch) warm-starts from. No benchmark inference here:
# a caption-only model has nothing to say on VSI-Bench/MMSI-Bench yet.
#
# Logs for the whole experiment: logs/pretrain_vjepa_sceneverse_sft/ (create it once
# before the first submission, Slurm opens the paths above before this script runs):
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft
#
# Prereqs:
#   sbatch dataset/download_open_vljepa_ckpt.sbatch     # -> ckpts/open_vljepa/best.pt
#   sbatch dataset/convert_sceneverse_pretrain.sbatch   # -> data/sceneverse_scene_pretrain_{train,val}.jsonl
#   sbatch fusion_gv/trainbatch/overfit_pretrain_vjepa_sceneverse_sft.sbatch   # plumbing check first
#
# --time=2-00:00:00 is a guess (there's no throughput number for this data yet); if it
# runs out, the step_*.pt checkpoints are on disk but train_gvjepa.py cannot resume from one.

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/pretrain_vjepa_sceneverse_sft"

cd "$ROOT"
mkdir -p "$LOG_DIR"

MONITOR_LOG="$LOG_DIR/pretrain_gpu_monitor_${SLURM_JOB_ID}.log"

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

echo "[pretrain] start $(date)"
$SIF accelerate launch --multi_gpu --num_processes 2 \
    fusion_gv/train_gvjepa.py \
    --config fusion_gv/configs/pretrain_vjepa_sceneverse.yaml
echo "[pretrain] done $(date)"
