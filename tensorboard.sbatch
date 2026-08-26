#!/bin/bash
#SBATCH --job-name=tensorboard
#SBATCH --partition=geo
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/tensorboard_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/tensorboard_%j.err

# View a torch.profiler trace in TensorBoard. Launch-only -- no install logic
# in this script. One-time env setup (a venv, not the plain container) is
# done interactively first:
#   srun --partition=geo --cpus-per-task=2 --mem=8G --time=00:30:00 --pty \
#       singularity exec --bind /glob/g01-cache:/glob/g01-cache \
#       /glob/g01-cache/container/nvpytorch24.05-py3.simg bash
#   python3 -m venv --system-site-packages /glob/g01-cache/pf/Yushuo/venv-tensorboard
#   source /glob/g01-cache/pf/Yushuo/venv-tensorboard/bin/activate
#   pip install torch_tb_profiler
#   python3 -c "import torch_tb_profiler; print('ok')"
#
# torch_tb_profiler is the correct plugin for torch.profiler traces --
# NOT tensorboard-plugin-profile, that's TensorFlow's unrelated profiler.
# --system-site-packages lets the venv see the container's own already-
# working numpy/PIL/etc instead of reinstalling them, so this pip install
# only pulls torch_tb_profiler's own (small) dependency chain.
#
# This file only uses absolute paths and `cd`s into the project root below
# -- its own location on disk doesn't matter, safe to move/rename freely.
#
# Log dir must exist BEFORE first submission (Slurm opens --output/--error
# before this script body runs):
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs
#
# Override LOGDIR to narrow which run's trace to show:
#   LOGDIR=outputs/sft_from_vljepa_ckpt_fusiongv_overfit/profiler_trace sbatch tensorboard.sbatch

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/container/nvpytorch24.05-py3.simg"
VENV="/glob/g01-cache/pf/Yushuo/venv-tensorboard"
LOGDIR="${LOGDIR:-outputs}"
PORT=6006

cd "$ROOT"

NODE=$(hostname)
echo "============================================"
echo "TensorBoard running on node: $NODE"
echo "Port: $PORT"
echo "Logdir: $LOGDIR"
echo ""
echo "SSH tunnel command (run on LOCAL machine):"
echo "  ssh -L ${PORT}:${NODE}:${PORT} ssh_bgd"
echo ""
echo "Then open: http://localhost:${PORT}"
echo "============================================"

srun singularity exec \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    $VENV/bin/python3 -m tensorboard.main --logdir "$LOGDIR" --port $PORT --bind_all
