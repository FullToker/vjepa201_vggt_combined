#!/bin/bash
#SBATCH --job-name=infer_spar_eval_old_sample9k
#SBATCH --partition=geo
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/infer_spar_eval_old_sample9k/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/infer_spar_eval_old_sample9k/%j.err

# Runs the old SPAR select-format eval against both open-vljepa-warm-started
# SFT checkpoints, one per GPU, in parallel. infer_gvjepa.py is single-process
# (no accelerate) -- CUDA_VISIBLE_DEVICES pins each run to its own GPU instead
# of splitting one run across both.
#
# Uses spar_sftqa_eval_old_3k.jsonl (pre-sampled subset) instead of the full
# manifest -- the full file was estimated at ~155h, too long to wait out, and
# a --time timeout would kill the job mid-run with no metrics.json at all
# (that file is only written after the full loop finishes -- see
# infer_gvjepa.py) plus a predictions.jsonl covering only whatever source/type
# happened to be first in file order (convert_spar_sftqa.py writes
# source-by-source, not shuffled).
#
# Requires both SFT runs to already have written final.pt:
#   fusion_gv/trainbatch/sft_from_vljepa_ckpt.sbatch          -> vjepa-only
#   fusion_gv/trainbatch/sft_from_vljepa_ckpt_fusiongv.sbatch -> VGGT+JEPA
#
# The #SBATCH --output/--error paths above are opened by Slurm before this
# script body runs, so mkdir -p inside the script is too late for THOSE two --
# create the folder once before first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/infer_spar_eval_old_sample9k

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EVAL_JSONL="/glob/g01-cache/pf/Yushuo/vjepa201/data/spar_sftqa_eval_old_3k.jsonl"

mkdir -p "$ROOT/logs/infer_spar_eval_old_sample9k"

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
        --output ./outputs/inference/spar_eval_old_vjepa_sample9k/predictions.jsonl \
        --metrics-output ./outputs/inference/spar_eval_old_vjepa_sample9k/metrics.json \
        --mode select &
pid_vjepa=$!

CUDA_VISIBLE_DEVICES=1 singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    python fusion_gv/infer_gvjepa.py \
        --config fusion_gv/configs/sft_from_vljepa_ckpt_fusiongv.yaml \
        --checkpoint ./outputs/sft_from_vljepa_ckpt_fusiongv/final.pt \
        --input "$EVAL_JSONL" \
        --output ./outputs/inference/spar_eval_old_fusiongv_sample9k/predictions.jsonl \
        --metrics-output ./outputs/inference/spar_eval_old_fusiongv_sample9k/metrics.json \
        --mode select &
pid_fusion_gv=$!

status=0
wait "$pid_vjepa"       || { echo "[infer_spar_eval_old] vjepa run failed"      >&2; status=1; }
wait "$pid_fusion_gv"   || { echo "[infer_spar_eval_old] fusion_gv run failed"  >&2; status=1; }
exit $status
