#!/bin/bash
#SBATCH --job-name=dl_embodiedscan_all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_embodiedscan_all_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_embodiedscan_all_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
mkdir -p "$ROOT/logs"

source "$ROOT/.env/hf.env"

srun bash "$ROOT/dataset/download_embodiedscan.sh" "$ROOT/source_data/embodiedscan"
srun bash "$ROOT/dataset/download_scannet_posed.sh" "$ROOT/source_data/scannet"

echo "All done at $(date)"
