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
PYUSERBASE="$ROOT/.local"
CONTAINER="/glob/g01-cache/container/nvpytorch24.05-py3.simg"
SIF="srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

mkdir -p "$ROOT/logs" "$PYUSERBASE"
source "$ROOT/.env/hf.env"

# ── 安装并检查依赖 ────────────────────────────────────────────────────
echo "Installing huggingface_hub..."
$SIF python3 -m pip install --user -q -U "huggingface_hub"

echo "Checking imports..."
$SIF python3 -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
$SIF python3 -c "import zipfile, pickle, json, pathlib; print('stdlib OK')"

# huggingface-cli → hf symlink (以防下载脚本找hf)
if $SIF bash -c "command -v huggingface-cli && ! command -v hf" 2>/dev/null; then
    $SIF bash -c "ln -sf \$(which huggingface-cli) $PYUSERBASE/bin/hf"
fi
$SIF hf --version
echo "All checks passed at $(date)"

# ── 下载 ──────────────────────────────────────────────────────────────
$SIF bash "$ROOT/dataset/download_embodiedscan.sh" "$ROOT/source_data/embodiedscan"
$SIF bash "$ROOT/dataset/download_scannet_posed.sh" "$ROOT/source_data/scannet"

echo "All done at $(date)"
