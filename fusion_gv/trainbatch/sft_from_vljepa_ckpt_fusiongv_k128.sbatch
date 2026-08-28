#!/bin/bash
#SBATCH --job-name=sft_from_vljepa_ckpt_fusiongv_k128
#SBATCH --partition=geo
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=7-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv/%j.err

# Sibling of sft_from_vljepa_ckpt_fusiongv.sbatch, K=128 VisualResampler
# (fusion_gv/configs/sft_from_vljepa_ckpt_fusiongv_k128.yaml) instead of the
# K=1/mean-pool default. Same LOG_DIR as that script (jobid in the filename
# keeps the two runs' logs apart even sharing the dir) -- but a SEPARATE
# train.output_dir/vsi_inference.output_dir/mmsi_inference.output_dir (see
# that yaml), so the two runs' checkpoints/eval outputs never collide.
#
# No --profile here -- this is a real training run, not a profiling run (see
# fusion_gv/k_sweep/ for that). fusion_gv/profiling.py's --profile is opt-in
# and off by default anyway, so this script doesn't need to do anything
# special to keep it off; just doesn't pass the flag.
#
# LOG_DIR must exist BEFORE first submission (Slurm opens --output/--error
# before this script body runs) -- already true if
# sft_from_vljepa_ckpt_fusiongv.sbatch has been run before; otherwise:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv
#
# Prereqs: same as sft_from_vljepa_ckpt_fusiongv.sbatch (open-vljepa ckpt,
# spar_sftqa_sftvljepa_uni8.jsonl, vsibench/mmsibench manifests).
#
# To re-run only the infer steps after training already finished, skip training:
#   RUN_TRAIN=0 sbatch fusion_gv/trainbatch/sft_from_vljepa_ckpt_fusiongv_k128.sbatch

set -euo pipefail
RUN_TRAIN=${RUN_TRAIN:-1}

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_from_vljepa_ckpt_fusiongv"
CONFIG="fusion_gv/configs/sft_from_vljepa_ckpt_fusiongv_k128.yaml"

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
        --config "$CONFIG"
    echo "[train] done $(date)"
else
    echo "[train] skipped (RUN_TRAIN=0), reusing existing checkpoint"
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
