#!/bin/bash
#SBATCH --job-name=extract_vsibench_frames_32f
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/extract_vsibench_frames_32f_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/extract_vsibench_frames_32f_%j.err

# 32-frame VSI-Bench extraction + manifest. Sibling of
# extract_vsibench_frames.sbatch (8 frames) -- separate frames dir
# (<vsi-dir>/frames32_fps/) and separate manifest
# (data/vsibench_manifest_32f.jsonl), so the 8-frame set stays intact.
#
# Motivation: the 8-frame manifest exercises the video path only; VGGT's
# multi-view geometry aggregation barely contributes at 8 sparse frames. 32
# frames is the most the arch allows (_S_MAX = 33 in fusion_gv/gvjepa.py).
#
# pool-size 128 (vs 64 for 8 frames): FPS needs a candidate pool comfortably
# larger than num_frames or the "spread" degrades toward uniform sampling.

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
SIF="srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

mkdir -p "$ROOT/logs"

echo "Start 32-frame extraction at $(date)"

$SIF bash "$ROOT/dataset/extract_vsibench_frames.sh" \
    "$ROOT/source_data/vsibench" \
    "$ROOT/data/vsibench_manifest_32f.jsonl" \
    8 \
    32 \
    128

echo "All done at $(date)"
