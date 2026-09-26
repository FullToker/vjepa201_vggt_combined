#!/bin/bash
#SBATCH --job-name=conv_sceneverse_pretrain
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft/convert_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft/convert_%j.err

# Builds data/sceneverse_scene_pretrain_{train,val}.jsonl for the stage-1 pretraining
# (fusion_gv/configs/pretrain_vjepa_sceneverse.yaml): 8-frame sets of one ScanNet/3RScan
# scan -> scene caption, with a space-level group_id and a space-level train/val split.
# See dataset/convert_sceneverse_pretrain.py for the sampling and split rules.
#
# Needs source_data/3rscan/3RScan.json (notebooks/rescan_check.ipynb fetches it).
# CPU only; the time is mostly listing ~2.9k frame directories on network storage.
#
# All logs of this experiment live in logs/pretrain_vjepa_sceneverse_sft/. Slurm opens the
# #SBATCH --output/--error paths before this script runs, so create the folder once
# before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/pretrain_vjepa_sceneverse_sft

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

mkdir -p "$ROOT/logs/pretrain_vjepa_sceneverse_sft" "$ROOT/data"

echo "[convert-sceneverse] start $(date)"
srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    python3 "$ROOT/dataset/convert_sceneverse_pretrain.py" --root "$ROOT"
echo "[convert-sceneverse] done $(date)"
