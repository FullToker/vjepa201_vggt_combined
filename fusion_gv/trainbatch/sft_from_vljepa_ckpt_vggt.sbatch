#!/bin/bash
#SBATCH --job-name=sft_from_vljepa_ckpt_vggt
#SBATCH --partition=geo
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_vggt/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_vggt/%j.err

# 2 GPUs (was 1): train, then VSI-Bench infer, then MMSI-Bench infer, all
# against fusion_gv/configs/sft_from_vljepa_ckpt_vggt.yaml. Third arm of the
# X-encoder ablation (vjepa / ijepa / vggt). batch_size=64 / accum=6 are
# unchanged from the single-GPU version -- only the GPU count moved, so
# effective batch doubled (384 -> 768) and max_steps was halved (6920 ->
# 3460) to keep the same ~1.5-epoch data coverage; see the yaml. Also now
# carries visual_pool_k=128 (was the default 1/mean-pool) -- a separate,
# also-unmeasured memory load on top of the GPU-count change.
#
# *** READ BEFORE SUBMITTING ***
# No combination of 2 GPUs / this batch_size/accum / K=128 has been profiled
# -- the overfit job (single GPU, same per-GPU batch_size and K) is what
# actually checks it fits, not this comment. train_gvjepa.py also has NO
# checkpoint-resume flag -- if the job hits --time before max_steps, the
# step_*.pt files in train.output_dir exist but nothing loads one back in to
# continue training from it.
#
# Same --mem=200G rationale as the vjepa/ijepa arms' real runs (prior
# host-RAM OOM at MaxRSS=155GB on a multi-day run) -- carried over
# defensively, not re-derived for this one.
#
# All logs live under logs/sft_from_vljepa_ckpt_vggt/. The #SBATCH
# --output/--error paths above are opened by Slurm before this script body
# runs, so mkdir -p inside the script is too late for THOSE two -- create
# the folder once before first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_vggt
#
# Prereqs (run once, before this job):
#   sbatch dataset/download_open_vljepa_ckpt.sbatch    # -> ckpts/open_vljepa/best.pt
#   python download_ckpts.py --models vggt              # -> ckpts/vggt.pt, if not already there
#   sbatch dataset/convert_spar_sftqa_uni8.sbatch       # -> data/spar_sftqa_sftvljepa_uni8.jsonl
#   dataset/make_vsibench_manifest.py / make_mmsibench_manifest.py already run # -> data/{vsibench,mmsibench}_manifest.jsonl
#
# Run the overfit check first -- it's also what tells you whether
# batch_size=64 + K=128 fits before committing a week of 2-GPU time to this:
#   sbatch fusion_gv/trainbatch/overfit_sft_from_vljepa_ckpt_vggt.sbatch
#
# To re-run only the infer steps after training already finished:
#   RUN_TRAIN=0 sbatch fusion_gv/trainbatch/sft_from_vljepa_ckpt_vggt.sbatch
# Requires train.output_dir's final.pt (or step_*.pt) to already exist.

set -euo pipefail
RUN_TRAIN=${RUN_TRAIN:-1}

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_from_vljepa_ckpt_vggt"

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
    $SIF accelerate launch --multi_gpu --num_processes 2 \
        fusion_gv/train_gvjepa.py \
        --config fusion_gv/configs/sft_from_vljepa_ckpt_vggt.yaml
    echo "[train] done $(date)"
else
    echo "[train] skipped (RUN_TRAIN=0), reusing existing checkpoint"
fi

echo "[infer-vsibench] start $(date)"
$SIF python3 fusion_gv/infer_vsibench.py \
    --config fusion_gv/configs/sft_from_vljepa_ckpt_vggt.yaml \
    --config-section vsi_inference
echo "[infer-vsibench] done $(date)"

echo "[infer-mmsibench] start $(date)"
$SIF python3 fusion_gv/infer_mmsibench.py \
    --config fusion_gv/configs/sft_from_vljepa_ckpt_vggt.yaml \
    --config-section mmsi_inference
echo "[infer-mmsibench] done $(date)"
