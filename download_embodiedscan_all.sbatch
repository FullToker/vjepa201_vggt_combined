#!/bin/bash
#SBATCH --job-name=dl_embodiedscan_all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_embodiedscan_all_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_embodiedscan_all_%j.err

set -euo pipefail

source /glob/g01-cache/pf/Yushuo/vjepa201/.env/hf.env

HF_BIN="/glob/g01-cache/pf/Yushuo/vjepa201/.local/bin"
CONTAINER="/glob/g01-cache/container/nvpytorch24.05-py3.simg"
SIF="srun singularity exec -H /glob/g01-cache:/glob/g01-cache --env PATH=$HF_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin $CONTAINER"

ANNO_DIR="/glob/g01-cache/pf/Yushuo/vjepa201/source_data/embodiedscan"
IMG_DIR="/glob/g01-cache/pf/Yushuo/vjepa201/source_data/scannet"
mkdir -p "$ANNO_DIR" "$IMG_DIR" /glob/g01-cache/pf/Yushuo/vjepa201/logs "$HF_BIN"

# ── 安装hf CLI到可写目录 ──────────────────────────────────────────────
echo "Installing hf CLI..."
$SIF bash -c "curl -LsSf https://hf.co/cli/install.sh | HF_INSTALL_DIR=$HF_BIN bash"

# ── 环境检查 ─────────────────────────────────────────────────────────
echo "Checking environment..."
$SIF python3 --version
$SIF hf --version
echo "Environment check passed at $(date)"

# ── 1. EmbodiedScan PKL标注 ──────────────────────────────────────────
echo "[1/2] Downloading EmbodiedScan annotations..."
$SIF hf download cjfcsjt/embodiedscan embodiedscan.zip \
    --repo-type dataset \
    --local-dir "$ANNO_DIR" \
    --token "$HF_TOKEN"

echo "Extracting annotations..."
$SIF python3 -m zipfile -e "${ANNO_DIR}/embodiedscan.zip" "${ANNO_DIR}"
rm "${ANNO_DIR}/embodiedscan.zip"
echo "Annotations done at $(date)"

# ── 2. ScanNet posed images (115GB, 11 parts) ────────────────────────
echo "[2/2] Downloading ScanNet posed images..."
PARTS=(
  annotated.tar.gz.part00
  annotated.tar.gz.part01
  annotated.tar.gz.part02
  annotated.tar.gz.part03
  annotated.tar.gz.part04
  annotated.tar.gz.part05
  annotated.tar.gz.part06
  annotated.tar.gz.part07
  annotated.tar.gz.part08
  annotated.tar.gz.part09
  annotated.tar.gz.part10
)

for part in "${PARTS[@]}"; do
    echo "  -> ${part}"
    $SIF hf download GUESSGUO/annotated_posed_images_scannet "$part" \
        --repo-type dataset \
        --local-dir "$IMG_DIR" \
        --token "$HF_TOKEN"
done

echo "Extracting ScanNet images..."
cat "${IMG_DIR}"/annotated.tar.gz.part* | tar -xzf - -C "$IMG_DIR"
rm "${IMG_DIR}"/annotated.tar.gz.part*

echo "All done at $(date)"