#!/bin/bash
#SBATCH --job-name=infer_spar_eval_old
#SBATCH --partition=geo
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/infer_spar_eval_old_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/infer_spar_eval_old_%j.err

# Runs the old SPAR select-format eval (spar_sftqa_eval_old.jsonl) against
# both open-vljepa-warm-started SFT checkpoints, one per GPU, in parallel.
# infer_gvjepa.py is single-process (no accelerate) -- CUDA_VISIBLE_DEVICES
# pins each run to its own GPU instead of splitting one run across both.
#
# Requires both SFT runs to already have written final.pt:
#   fusion_gv/trainbatch/sft_from_vljepa_ckpt.sbatch            -> vjepa-only
#   fusion_gv/trainbatch/sft_from_vljepa_ckpt_fusion_gv.sbatch  -> VGGT+JEPA

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EVAL_JSONL="/glob/g01-cache/pf/Yushuo/vjepa201/data/spar_sftqa_eval_old.jsonl"

mkdir -p "$ROOT/logs"

CUDA_VISIBLE_DEVICES=0 singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    python fusion_gv/infer_gvjepa.py \
        --config fusion_gv/configs/sft_from_vljepa_ckpt.yaml \
        --checkpoint ./outputs/sft_from_vljepa_ckpt/final.pt \
        --input "$EVAL_JSONL" \
        --output ./outputs/inference/spar_eval_old_vjepa/predictions.jsonl \
        --metrics-output ./outputs/inference/spar_eval_old_vjepa/metrics.json \
        --mode select &
pid_vjepa=$!

CUDA_VISIBLE_DEVICES=1 singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    python fusion_gv/infer_gvjepa.py \
        --config fusion_gv/configs/sft_from_vljepa_ckpt_fusion_gv.yaml \
        --checkpoint ./outputs/sft_from_vljepa_ckpt_fusion_gv/final.pt \
        --input "$EVAL_JSONL" \
        --output ./outputs/inference/spar_eval_old_fusion_gv/predictions.jsonl \
        --metrics-output ./outputs/inference/spar_eval_old_fusion_gv/metrics.json \
        --mode select &
pid_fusion_gv=$!

status=0
wait "$pid_vjepa"       || { echo "[infer_spar_eval_old] vjepa run failed"      >&2; status=1; }
wait "$pid_fusion_gv"   || { echo "[infer_spar_eval_old] fusion_gv run failed"  >&2; status=1; }
exit $status
