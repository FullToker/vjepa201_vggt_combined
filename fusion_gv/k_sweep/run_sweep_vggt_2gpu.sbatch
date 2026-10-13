#!/bin/bash
#SBATCH --job-name=k_sweep_vggt_2gpu
#SBATCH --partition=geo
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/k_sweep/vggt_2gpu_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/k_sweep/vggt_2gpu_%j.err

# VGGT-only (x_encoder_type=vggt), 2 GPUs: how large can visual_pool_k go
# before OOM. Same launcher as run_sweep.sbatch, see run_sweep.py for the
# sweep logic. Per-GPU batch is BATCH below (default 64 = train.batch_size
# in sft_from_vljepa_ckpt_vggt.yaml), so the numbers transfer to the real run.
# Stops at the first K that fails. Result table at the end of the .out, and:
#   outputs/k_sweep_vggt_2gpu/k_sweep_results.json
#   outputs/k_sweep_vggt_2gpu/k<K>/stderr.log      (traceback of a failed K)
# Log dir must exist before first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/vjepa201/logs/k_sweep
#
# Other K values / batch size:
#   K_VALUES=128,160,192 BATCH=32 sbatch fusion_gv/k_sweep/run_sweep_vggt_2gpu.sbatch

set -euo pipefail
K_VALUES=${K_VALUES:-32,64,96,128,192,256,384,512}
BATCH=${BATCH:-64}

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER \
    python3 fusion_gv/k_sweep/run_sweep.py \
        --config fusion_gv/configs/overfit_from_vljepa_ckpt_vggt.yaml \
        --out-root outputs/k_sweep_vggt_2gpu \
        --num-gpus 2 \
        --overfit-samples "$BATCH" \
        --k-values "$K_VALUES" \
        --stop-on-fail
