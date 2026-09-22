#!/bin/bash
#SBATCH --job-name=dl_sceneverse
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_sceneverse_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_sceneverse_%j.err

# Downloads SceneVerse's ScanNet scene_cap.json + 3RScan.zip from Google
# Drive straight onto cluster storage (dataset/download_sceneverse.sh --
# see that file for the resourcekey/permission caveats). Lands in
# source_data/sceneverse/, alongside the existing source_data/{scannet,3rscan}
# raw scans these captions are keyed against.
#
# gdown isn't in the container image -- pip install --user puts it under
# $ROOT (bound as HOME via -H below), so it persists across runs instead of
# reinstalling every job.
#
# --time=02:00:00: 3RScan.zip's size is unknown (that's exactly what this
# job's `unzip -l` at the end is for) -- padded generously since Drive
# downloads can be slow/throttled; bump if it times out mid-download.

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"

srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    bash -c '
        set -euo pipefail
        command -v gdown >/dev/null 2>&1 || pip install --user -q gdown

        cd "'"$ROOT"'"

        echo "[dl-sceneverse] 3rscan.zip start $(date)"
        bash dataset/download_sceneverse.sh \
            18PFyrvSyD0jbFDurq9j4Ft_nYHtuaGez 3rscan.zip ./source_data/sceneverse
        echo "[dl-sceneverse] 3rscan.zip done $(date)"

        echo "[dl-sceneverse] scannet_scene_cap.json start $(date)"
        bash dataset/download_sceneverse.sh \
            1c6SdqTPUoCMeY4mpVLEjPROtGDlz24oX scannet_scene_cap.json ./source_data/sceneverse
        echo "[dl-sceneverse] scannet_scene_cap.json done $(date)"
    '

echo "All done at $(date)"
