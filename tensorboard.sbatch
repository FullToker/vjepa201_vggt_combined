#!/bin/bash
#SBATCH --job-name=tensorboard
#SBATCH --partition=geo
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/tensorboard_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/tensorboard_%j.err

# View a torch.profiler trace (from fusion_gv/trainbatch/profile_gvjepa.sbatch
# or any --profile run) in TensorBoard. Same SSH-tunnel pattern as
# jupyter.sbatch. CPU-only, no --gres=gpu needed.
#
# Log dir must exist BEFORE first submission (Slurm opens --output/--error
# before this script body runs, mkdir -p below is too late for those two):
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs
#
# Default LOGDIR below points at the whole outputs/ tree so both the train
# trace (outputs/<run>/profiler_trace) and the infer trace
# (outputs/inference/<run>/.../profiler_trace) show up as separate runs in
# the same TensorBoard instance. Override to narrow it:
#   LOGDIR=outputs/sft_from_vljepa_ckpt_fusiongv_overfit/profiler_trace sbatch tensorboard.sbatch

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOGDIR="${LOGDIR:-outputs}"
PORT=6006

mkdir -p "$ROOT/logs"

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
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    tensorboard --logdir "$LOGDIR" --port $PORT --bind_all
