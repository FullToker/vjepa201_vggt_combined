#!/bin/bash
#SBATCH --job-name=test_forward_ijepa
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/test_forward_ijepa/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/test_forward_ijepa/%j.err

# Validates the new x_encoder_type="ijepa" ablation path end-to-end
# (fusion_gv/test_forward.py Phase 6): downloads the official I-JEPA
# ViT-H/14 checkpoint (IN1K-vit.h.14-300e.pth.tar, ~9.65GB -- a full
# training checkpoint bundling encoder/target_encoder/predictor/optimizer
# state; only target_encoder gets loaded, see fusion_gv/encoders.py
# FrozenIJEPA), then runs the full test_forward.py suite (Phases 1-6) so
# regressions in the existing fusion_gv/vjepa paths are caught too.
#
# --time=00:45:00: generous vs. the ~2-3min a training node's bandwidth
# should need for 9.65GB -- padded because this is the first run and
# compute-node throughput to dl.fbaipublicfiles.com is untested here.
#
# Checkpoint is NOT deleted at the end -- unlike the throwaway local/docker
# verification run, this one lives in $ROOT/ckpts so later ijepa-ablation
# training runs (sft configs with x_encoder_type: ijepa) don't have to
# re-pull 9.65GB. Delete manually if you only wanted the one-off check:
#   rm $ROOT/ckpts/IN1K-vit.h.14-300e.pth.tar

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs/test_forward_ijepa"

SIF="srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

echo "[test-forward-ijepa] download start $(date)"
$SIF python3 download_ckpts.py --models ijepa
echo "[test-forward-ijepa] download done  $(date)"

echo "[test-forward-ijepa] test start $(date)"
$SIF python3 fusion_gv/test_forward.py
echo "[test-forward-ijepa] test done  $(date)"
