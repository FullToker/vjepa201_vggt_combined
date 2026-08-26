#!/bin/bash
#SBATCH --job-name=k_sweep
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/k_sweep/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/k_sweep/%j.err

# Launch-only, no logic here -- see fusion_gv/k_sweep/run_sweep.py for the
# actual sweep (loops K, runs, records, continues past OOM). One result file:
#   outputs/k_sweep/k_sweep_results.json
# Log dir must exist before first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/k_sweep

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    python3 fusion_gv/k_sweep/run_sweep.py
