#!/bin/bash
#SBATCH --job-name=inspect_escan
#SBATCH --partition=geo
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/inspect_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/inspect_%j.err

srun singularity exec \
    --bind /glob/g01-cache:/glob/g01-cache \
    /glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif \
    python3 /glob/g01-cache/pf/Yushuo/vjepa201/inspect.py
