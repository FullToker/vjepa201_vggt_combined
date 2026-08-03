#!/bin/bash
#SBATCH --job-name=conv_spar_uni8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/conv_spar_uni8_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/conv_spar_uni8_%j.err

# Builds spar_sftqa_sftvljepa_uni8.jsonl for fusion_gv/configs/sft_from_vljepa_ckpt.yaml.
#
# --types restricts to the 6 SPAR types that are BOTH qualitative (existence/
# attribute/relation, no metric distance/depth/room-size numbers) AND not
# single-image (real source image count checked via the ad-hoc stats script
# earlier: these 6 have 2-3 distinct real frames; the rest of the 24 SPAR
# types are either single-image (obj_spatial_relation_oo, spatial_imagination_oc,
# spatial_imagination_oo) or metric (depth_prediction_*, distance_prediction_*,
# distance_infer_center_*, room_size) and are excluded on purpose:
#   camera_motion_infer          (2 real frames)
#   position_matching            (2 real frames)
#   obj_spatial_relation_oc_mv   (3 real frames)
#   obj_spatial_relation_oo_mv   (3 real frames)
#   spatial_imagination_oc_mv    (3 real frames)
#   spatial_imagination_oo_mv    (3 real frames)
#
# --num-frames 8 still pads/cycles each row up to 8 (via sample_frames()'s
# cycle-fill for rows shorter than num_frames) so these mixed 2-frame/3-frame
# types can batch together at a fixed S=8 — same mechanism spar_sftqa_train.jsonl
# already relies on, just restricted to these 6 types instead of all 24.
#
# Train-only: --only train (this stage doesn't need the eval/select manifest).

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
SIF="srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

mkdir -p "$ROOT/logs" "$ROOT/data"

echo "[spar-uni8] convert start $(date)"
$SIF python3 "$ROOT/dataset/convert_spar_sftqa.py" \
    --spar-root "$ROOT/source_data/spar" \
    --output-dir "$ROOT/data" \
    --train-output spar_sftqa_sftvljepa_uni8.jsonl \
    --types camera_motion_infer position_matching \
             obj_spatial_relation_oc_mv obj_spatial_relation_oo_mv \
             spatial_imagination_oc_mv spatial_imagination_oo_mv \
    --num-frames 8 \
    --only train
echo "[spar-uni8] done $(date)"
