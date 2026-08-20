#!/bin/bash
#SBATCH --job-name=dl_matterport3d
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_matterport3d_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/dl_matterport3d_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs"

# FILE_TYPES defaults to undistorted_color_images + undistorted_camera_parameters
# + region_segmentations (posed RGB + region annotation, what MMScan needs) --
# NOT the full 17 file types (~1.3TB). Override e.g.:
#   FILE_TYPES="undistorted_color_images undistorted_depth_images" sbatch dataset/download_matterport3d.sbatch
FILE_TYPES="${FILE_TYPES:-undistorted_color_images undistorted_camera_parameters region_segmentations}"

echo "Start download at $(date)"

srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    bash "$ROOT/dataset/download_matterport3d.sh" \
        "$ROOT/source_data/matterport3d" \
        $FILE_TYPES

echo "All done at $(date)"
