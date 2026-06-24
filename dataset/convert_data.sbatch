#!/bin/bash
#SBATCH --job-name=conv_data
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/conv_data_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/conv_data_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
SIF="srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

mkdir -p "$ROOT/logs" "$ROOT/data"

# ── EmbodiedScan train ────────────────────────────────────────────────────────
CONVERT_ESCAN=${CONVERT_ESCAN:-1}
CONVERT_SPAR=${CONVERT_SPAR:-1}

if [ "$CONVERT_ESCAN" = "1" ]; then
    echo "[escan] EmbodiedScan convert start $(date)"
    $SIF python3 "$ROOT/dataset/convert_embodiedscan.py" \
        --split train \
        --data-root "$ROOT/source_data" \
        --output "$ROOT/data/embodiedscan_train.jsonl" \
        --max-views 4
    echo "[escan] EmbodiedScan done $(date)"
fi

# ── SPAR-7M SFT QA (train + eval) ────────────────────────────────────────────
if [ "$CONVERT_SPAR" = "1" ]; then
    echo "[spar] SPAR convert start $(date)"
    $SIF python3 "$ROOT/dataset/convert_spar_sftqa.py" \
        --spar-root "$ROOT/source_data/spar" \
        --output-dir "$ROOT/data" \
        --num-frames 4 \
        --only both
    echo "[spar] SPAR done $(date)"
fi

echo "All done $(date)"
