#!/bin/bash
#SBATCH --job-name=sft_from_vljepa_ckpt
#SBATCH --partition=geo
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt/%j.err

# 3 GPUs, single job: train, then VSI-Bench infer, then MMSI-Bench infer,
# all against fusion_gv/configs/sft_from_vljepa_ckpt.yaml (train + vsi_inference
# + mmsi_inference sections all live in that one file, including max_steps
# recomputed for the 3x larger global batch -- see that yaml's train: section).
# Infer steps read the checkpoint train just wrote via train.output_dir ->
# final.pt or the latest step_*.pt (see infer_gvjepa._latest_checkpoint), no
# path to fill in between steps.
#
# --mem=200G (was 150G): a prior 1-GPU run OOM'd on host RAM at MaxRSS=155GB
# after ~3d20h (sacct -j <jobid> confirmed State=FAILED, MaxRSS > --mem). This
# bumps the ceiling but doesn't fix the underlying slow leak -- if it OOMs
# again at a higher MaxRSS, the leak needs actual profiling, not just more
# --mem headroom.
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
#
# To re-run only the infer steps after training already finished (e.g. eval
# on an updated benchmark manifest, or after fixing an infer_vsibench.py/
# infer_mmsibench.py bug), skip training:
#   RUN_TRAIN=0 sbatch fusion_gv/trainbatch/sft_from_vljepa_ckpt.sbatch
# Requires train.output_dir's final.pt (or step_*.pt) to already exist.

set -euo pipefail
RUN_TRAIN=${RUN_TRAIN:-1}

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

if [ "$RUN_TRAIN" = "1" ]; then
    echo "[train] start $(date)"
    $SIF accelerate launch --multi_gpu --num_processes 3 \
        fusion_gv/train_gvjepa.py \
        --config fusion_gv/configs/sft_from_vljepa_ckpt.yaml
    echo "[train] done $(date)"
else
    echo "[train] skipped (RUN_TRAIN=0), reusing existing checkpoint"
fi

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
