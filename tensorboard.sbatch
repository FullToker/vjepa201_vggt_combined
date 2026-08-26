#!/bin/bash
#SBATCH --job-name=tensorboard
#SBATCH --partition=geo
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/tensorboard_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/tensorboard_%j.err

# View a torch.profiler trace in TensorBoard. Launch-only -- no install
# logic here. One-time env setup (venv + tensorboard-plugin-profile) is done
# interactively first, see the chat for the exact commands:
#   srun --partition=geo --cpus-per-task=2 --mem=8G --time=01:00:00 --pty \
#       singularity exec --bind /glob/g01-cache:/glob/g01-cache \
#       /glob/g01-cache/container/nvpytorch24.05-py3.simg bash
#   python3 -m venv --system-site-packages /glob/g01-cache/pf/Yushuo/venv-tensorboard
#   source /glob/g01-cache/pf/Yushuo/venv-tensorboard/bin/activate
#   pip install tensorboard-plugin-profile
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
    $VENV/bin/tensorboard --logdir "$LOGDIR" --port $PORT --bind_all
