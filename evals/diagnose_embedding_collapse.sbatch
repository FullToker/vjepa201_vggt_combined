#!/bin/bash
#SBATCH --job-name=diag_collapse
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/diag_collapse_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/diag_collapse_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"

# CONFIG/MANIFEST default to the MMSI-Bench run; override via env vars to
# point at the VSI-Bench config/manifest instead for the cross-check, e.g.:
#   CONFIG=fusion_gv/configs/infer_vsibench_mixedsft_8select.yaml \
#   MANIFEST=$ROOT/data/vsibench_manifest.jsonl \
#   sbatch evals/diagnose_embedding_collapse.sbatch
CONFIG="${CONFIG:-fusion_gv/configs/infer_mmsibench.yaml}"
MANIFEST="${MANIFEST:-$ROOT/data/mmsibench_manifest.jsonl}"

srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    python3 evals/diagnose_embedding_collapse.py \
        --config "$CONFIG" \
        --manifest "$MANIFEST" \
        --num-samples 24
