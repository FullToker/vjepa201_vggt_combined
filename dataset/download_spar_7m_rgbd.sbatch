#!/bin/bash
#SBATCH --job-name=dl_spar_7m_rgbd
#SBATCH --partition=data
#SBATCH --qos=students_opportunistic
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/home/%u/logs/dl_spar_7m_rgbd_%j.out
#SBATCH --error=/mnt/home/%u/logs/dl_spar_7m_rgbd_%j.err

set -euo pipefail

# TODO: set to your actual project folder on this cluster
PROJECT_DIR="/mnt/projects/<your-project>"
REPO_ROOT="$PROJECT_DIR/vjepa2"
DEST_DIR="$PROJECT_DIR/source_data"

# ulm/jena have 5TB local /tmp -- download+extract there first, then copy to
# network storage. Avoids stale-file-handle errors from writing many small
# files directly over the network mount (per cluster wiki).
TMP_DIR="/tmp/spar_7m_rgbd_${SLURM_JOB_ID}"

mkdir -p "$TMP_DIR" "$DEST_DIR" "/mnt/home/${USER}/logs"

echo "Start download at $(date) on $(hostname)"

bash "$REPO_ROOT/dataset/download_spar_7m_rgbd.sh" \
    --output-dir "$TMP_DIR" \
    --extract \
    --delete-archives

echo "Copying extracted data to network storage: $DEST_DIR"
cp -r "$TMP_DIR"/. "$DEST_DIR"/
rm -rf "$TMP_DIR"

echo "All done at $(date)"
