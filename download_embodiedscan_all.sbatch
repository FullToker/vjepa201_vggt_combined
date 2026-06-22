#!/bin/bash
  #SBATCH --job-name=dl_embodiedscan_all
  #SBATCH --cpus-per-task=4
  #SBATCH --mem=32G
  #SBATCH --time=48:00:00
  #SBATCH --output=/glob/g01-cache/pf/Yushuo/pf/vjepa201/logs/dl_embodiedscan_all_%j.out
  #SBATCH --error=/glob/g01-cache/pf/Yushuo/pf/vjepa201/logs/dl_embodiedscan_all_%j.err

  set -euo pipefail

  source /glob/g01-cache/pf/Yushuo/pf/vjepa201/.env/hf.env

  ANNO_DIR="/glob/g01-cache/pf/Yushuo/pf/vjepa201/source_data/embodiedscan"
  IMG_DIR="/glob/g01-cache/pf/Yushuo/pf/vjepa201/source_data/scannet"
  mkdir -p "$ANNO_DIR" "$IMG_DIR" /glob/g01-cache/pf/Yushuo/pf/vjepa201/logs

  pip install -q huggingface_hub[cli]

  # ── 1. EmbodiedScan PKL标注 ──────────────────────────────────────────
  echo "[1/2] Downloading EmbodiedScan annotations..."
  hf download cjfcsjt/embodiedscan embodiedscan.zip \
      --repo-type dataset \
      --local-dir "$ANNO_DIR" \
      --token "$HF_TOKEN"

  echo "Extracting annotations..."
  python3 -m zipfile -e "${ANNO_DIR}/embodiedscan.zip" "${ANNO_DIR}"
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
      hf download GUESSGUO/annotated_posed_images_scannet "$part" \
          --repo-type dataset \
          --local-dir "$IMG_DIR" \
          --token "$HF_TOKEN"
  done

  echo "Extracting ScanNet images..."
  cat "${IMG_DIR}"/annotated.tar.gz.part* | tar -xzf - -C "$IMG_DIR"
  rm "${IMG_DIR}"/annotated.tar.gz.part*

  echo "All done at $(date)"