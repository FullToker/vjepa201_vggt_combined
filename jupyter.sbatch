#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=geo
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/jupyter_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/jupyter_%j.err

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
PORT=8888

mkdir -p "$ROOT/logs" "$ROOT/notebooks"

NODE=$(hostname)
echo "============================================"
echo "Jupyter running on node: $NODE"
echo "Port: $PORT"
echo ""
echo "SSH tunnel command (run on LOCAL machine):"
echo "  ssh -L ${PORT}:${NODE}:${PORT} ssh_bgd"
echo ""
echo "Then open: http://localhost:${PORT}"
echo "============================================"

srun singularity exec \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    jupyter notebook \
        --no-browser \
        --port=$PORT \
        --ip=0.0.0.0 \
        --notebook-dir="$ROOT/notebooks" \
        --NotebookApp.token='' \
        --NotebookApp.password=''
