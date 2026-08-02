#!/bin/bash
#SBATCH --job-name=dl_open_vljepa_ckpt
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_open_vljepa_ckpt_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_open_vljepa_ckpt_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"

# Project-root ckpts dir — matches fusion_gv/configs/sft_from_vljepa_ckpt.yaml's
# init_vljepa_ckpt: ./ckpts/open_vljepa/best.pt
srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    bash "$ROOT/dataset/download_open_vljepa_ckpt.sh" \
        "$ROOT/ckpts/open_vljepa"

echo "All done at $(date)"
