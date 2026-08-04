#!/bin/bash
#SBATCH --job-name=sft_from_vljepa_ckpt_fusiongv
#SBATCH --partition=geo
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=7-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv/%j.err

# Sibling of sft_from_vljepa_ckpt.sbatch (x_encoder_type=vjepa, 1 GPU) — this
# one is fusion_gv (VGGT+JEPA concat, heavier), 3 GPUs, against
# fusion_gv/configs/sft_from_vljepa_ckpt_fusiongv.yaml. Single job: train ->
# VSI-Bench infer -> MMSI-Bench infer, all against that one yaml. See that
# yaml's header for the vis_proj shape-mismatch note (open-vljepa's checkpoint
# never saw the 2048-d fused encoder, so vis_proj stays random-init here,
# unlike the vjepa-only run where it transfers).
#
# All logs for this experiment live under logs/sft_from_vljepa_ckpt_fusiongv/
# instead of flat in logs/. The #SBATCH --output/--error paths above are opened
# by Slurm before this script body runs, so mkdir -p inside the script is too
# late for THOSE two -- create the folder once before first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv
#
# Prereqs (run once, before this job):
#   sbatch dataset/download_open_vljepa_ckpt.sbatch    # -> ckpts/open_vljepa/best.pt (shared with the vjepa-only run)
#   sbatch dataset/convert_spar_sftqa_uni8.sbatch       # -> data/spar_sftqa_sftvljepa_uni8.jsonl (shared)
#   dataset/make_vsibench_manifest.py / make_mmsibench_manifest.py already run # -> data/{vsibench,mmsibench}_manifest.jsonl
#   sbatch fusion_gv/trainbatch/overfit_sft_from_vljepa_ckpt_fusiongv.sbatch   # OOM/correctness check first
#
# To re-run only the infer steps after training already finished, skip training:
#   RUN_TRAIN=0 sbatch fusion_gv/trainbatch/sft_from_vljepa_ckpt_fusiongv.sbatch
# Requires train.output_dir's final.pt (or step_*.pt) to already exist.

set -euo pipefail
RUN_TRAIN=${RUN_TRAIN:-1}

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_from_vljepa_ckpt_fusiongv"

mkdir -p "$LOG_DIR"

MONITOR_LOG="$LOG_DIR/gpu_monitor_${SLURM_JOB_ID}.log"

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

if [ "$RUN_TRAIN" = "1" ]; then
    echo "[train] start $(date)"
    $SIF accelerate launch --multi_gpu --num_processes 3 \
        fusion_gv/train_gvjepa.py \
        --config fusion_gv/configs/sft_from_vljepa_ckpt_fusiongv.yaml
    echo "[train] done $(date)"
else
    echo "[train] skipped (RUN_TRAIN=0), reusing existing checkpoint"
fi

# Infer steps are single-GPU (infer_vsibench.py/infer_mmsibench.py don't use
# accelerate) -- fine to run on 1 of the 3 allocated GPUs after training ends.
echo "[infer-vsibench] start $(date)"
$SIF python3 fusion_gv/infer_vsibench.py \
    --config fusion_gv/configs/sft_from_vljepa_ckpt_fusiongv.yaml \
    --config-section vsi_inference
echo "[infer-vsibench] done $(date)"

echo "[infer-mmsibench] start $(date)"
$SIF python3 fusion_gv/infer_mmsibench.py \
    --config fusion_gv/configs/sft_from_vljepa_ckpt_fusiongv.yaml \
    --config-section mmsi_inference
echo "[infer-mmsibench] done $(date)"
