#!/bin/bash
#SBATCH --job-name=mmsibench_eval
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/mmsibench_eval_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/mmsibench_eval_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs" "$ROOT/data"

MANIFEST="$ROOT/data/mmsibench_manifest.jsonl"
if [[ -s "$MANIFEST" ]]; then
    echo "==> Manifest already exists, skipping build: $MANIFEST"
else
    echo "==> Building manifest $(date)"
    srun singularity exec \
        -H $ROOT:/home/go98qit \
        --bind /glob/g01-cache:/glob/g01-cache \
        $CONTAINER \
        python3 "$ROOT/dataset/make_mmsibench_manifest.py" \
            --raw-jsonl "$ROOT/data/mmsibench_raw.jsonl" \
            --out "$MANIFEST"
fi

echo "==> Running inference $(date)"
srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    python3 fusion_gv/infer_mmsibench.py \
        --config fusion_gv/configs/infer_mmsibench.yaml \
        --config-section inference

echo "All done $(date)"
