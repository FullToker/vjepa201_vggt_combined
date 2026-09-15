#!/bin/bash
#SBATCH --job-name=overfit_sft_vljepa_vggt
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_vggt/overfit_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_vggt/overfit_%j.err

# Single-GPU sanity check before the real sft_from_vljepa_ckpt_vggt.sbatch
# (also single-GPU, but a multi-day gradient-accumulation run) -- see that
# yaml's header for why this arm stays single-GPU instead of 3 like the
# vjepa/ijepa arms.
#   - freezes one batch (64 samples, matches train.batch_size=64 in that
#     yaml so per-GPU tensor shapes/memory match the real run) and repeats it
#   - 200 steps is enough to see loss drop AND to know within ~minutes
#     whether batch_size=64 OOMs -- VGGT (518px, ~1B params) has no prior
#     single-GPU profiling running alone (only ever measured fused with
#     V-JEPA), so 64 is a reasoned-but-unmeasured guess this run exists to
#     confirm -- if it comfortably fits, consider raising batch_size further
#     (and lowering gradient_accumulation_steps to keep effective batch 384)
#     in the real run's yaml before submitting it
#   - the infer steps then confirm the whole train->checkpoint->infer chain
#     runs end-to-end without crashing (accuracy numbers from an overfit
#     checkpoint are meaningless, this is a plumbing check, not an eval)
# Not a throughput benchmark -- _build_overfit_loader repeats the same fixed
# batch, so step/sec here does not predict the real run's dataloader-bound
# throughput.
#
# Own logs/sft_from_vljepa_ckpt_vggt/ dir (overfit_ prefix keeps filenames
# apart from the real run's) -- the #SBATCH --output/--error paths above are
# opened by Slurm before this script body runs, so mkdir -p inside the
# script is too late for those two -- create the folder once before first
# submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_vggt
#
# Prereqs (run once, before this job):
#   sbatch dataset/download_open_vljepa_ckpt.sbatch    # -> ckpts/open_vljepa/best.pt
#   python download_ckpts.py --models vggt              # -> ckpts/vggt.pt, if not already there
#   sbatch dataset/convert_spar_sftqa_uni8.sbatch       # -> data/spar_sftqa_sftvljepa_uni8.jsonl
#
# To re-run only the infer steps against an existing overfit checkpoint:
#   RUN_TRAIN=0 sbatch fusion_gv/trainbatch/overfit_sft_from_vljepa_ckpt_vggt.sbatch
# Requires ./outputs/sft_from_vljepa_ckpt_vggt_overfit/final.pt to already
# exist (train.output_dir in overfit_from_vljepa_ckpt_vggt.yaml).

set -euo pipefail
RUN_TRAIN=${RUN_TRAIN:-1}

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_from_vljepa_ckpt_vggt"
CONFIG="fusion_gv/configs/overfit_from_vljepa_ckpt_vggt.yaml"

mkdir -p "$LOG_DIR"

MONITOR_LOG="$LOG_DIR/gpu_monitor_overfit_${SLURM_JOB_ID}.log"

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
    echo "[overfit-train] start $(date)"
    $SIF accelerate launch --num_processes 1 \
        fusion_gv/train_gvjepa.py \
        --config "$CONFIG" \
        --overfit \
        --overfit-samples 64 \
        --overfit-steps 200
    echo "[overfit-train] done $(date)"
else
    echo "[overfit-train] skipped (RUN_TRAIN=0), reusing existing checkpoint"
fi

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
