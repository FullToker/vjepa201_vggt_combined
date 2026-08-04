#!/bin/bash
#SBATCH --job-name=sft_from_vljepa_ckpt
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=7-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt/%j.err

# Single-GPU, single job: train, then VSI-Bench infer, then MMSI-Bench infer,
# all against fusion_gv/configs/sft_from_vljepa_ckpt.yaml (train + vsi_inference
# + mmsi_inference sections all live in that one file). Infer steps read the
# checkpoint train just wrote via train.output_dir -> final.pt (save_every=0,
# see infer_gvjepa._latest_checkpoint), no path to fill in between steps.
#
# All logs for this experiment (Slurm stdout/stderr above, gpu/mem monitor
# below) live under logs/sft_from_vljepa_ckpt/ instead of flat in logs/.
# The #SBATCH --output/--error paths above are opened by Slurm before this
# script body runs, so mkdir -p inside the script is too late for THOSE two --
# create the folder once before first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt
#
# Prereqs (run once, before this job):
#   sbatch dataset/download_open_vljepa_ckpt.sbatch    # -> ckpts/open_vljepa/best.pt
#   sbatch dataset/convert_spar_sftqa_uni8.sbatch       # -> data/spar_sftqa_sftvljepa_uni8.jsonl
#   dataset/make_vsibench_manifest.py / make_mmsibench_manifest.py already run # -> data/{vsibench,mmsibench}_manifest.jsonl
#   sbatch fusion_gv/trainbatch/overfit_sft_from_vljepa_ckpt.sbatch   # OOM/correctness check first

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_from_vljepa_ckpt"

mkdir -p "$LOG_DIR"

MONITOR_LOG="$LOG_DIR/gpu_monitor_${SLURM_JOB_ID}.log"

nvidia-smi dmon -s pucvmet -d 30 > "$MONITOR_LOG" 2>&1 &

# Host RAM alongside GPU stats in the same log — this job OOM'd on host RAM
# before (DataLoader workers killed), not GPU memory, so track both.
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

echo "[train] start $(date)"
$SIF accelerate launch --num_processes 1 \
    fusion_gv/train_gvjepa.py \
    --config fusion_gv/configs/sft_from_vljepa_ckpt.yaml
echo "[train] done $(date)"

echo "[infer-vsibench] start $(date)"
$SIF python3 fusion_gv/infer_vsibench.py \
    --config fusion_gv/configs/sft_from_vljepa_ckpt.yaml \
    --config-section vsi_inference
echo "[infer-vsibench] done $(date)"

echo "[infer-mmsibench] start $(date)"
$SIF python3 fusion_gv/infer_mmsibench.py \
    --config fusion_gv/configs/sft_from_vljepa_ckpt.yaml \
    --config-section mmsi_inference
echo "[infer-mmsibench] done $(date)"
