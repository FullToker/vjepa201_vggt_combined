#!/bin/bash
#SBATCH --job-name=dl_spar_7m
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_spar_7m_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_spar_7m_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/container/nvpytorch24.05-py3.simg"
SIF="srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

mkdir -p "$ROOT/logs"

# wget在容器里检查
echo "Checking environment..."
$SIF wget --version | head -1
echo "Environment check passed at $(date)"

$SIF bash "$ROOT/vjepa2/dataset/download_spar_7m_rgbd.sh" \
    --output-dir "$ROOT/source_data" \
    --extract \
    --delete-archives

echo "All done at $(date)"
