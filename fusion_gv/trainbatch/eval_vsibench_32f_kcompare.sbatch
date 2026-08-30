#!/bin/bash
#SBATCH --job-name=eval_vsibench_32f_kcompare
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/vjepa201/logs/sft_from_vljepa_ckpt_fusiongv/%j.err

# Eval-only. Re-run VSI-Bench for BOTH already-trained fusion_gv checkpoints
# (K=1 mean-pool and K=128 VisualResampler) against the 32-frame manifest
# produced by dataset/extract_vsibench_frames_32f.sbatch, then diff the two.
#
#   K=1   ckpt : outputs/sft_from_vljepa_ckpt_fusiongv/final.pt        (config: sft_from_vljepa_ckpt_fusiongv.yaml)
#   K=128 ckpt : outputs/sft_from_vljepa_ckpt_fusiongv_k128/final.pt   (config: sft_from_vljepa_ckpt_fusiongv_k128.yaml)
#
# Outputs (separate dirs, never collide with the 8-frame runs):
#   outputs/inference/sft_from_vljepa_ckpt_fusiongv/vsibench_32f/vsibench_metrics.json
#   outputs/inference/sft_from_vljepa_ckpt_fusiongv_k128/vsibench_32f/vsibench_metrics.json
#
# CAVEATS
#  * frame_embed (fusion_gv/gvjepa.py) is nn.Embedding(_S_MAX=33); rows 8..32
#    never received a gradient (training ran num_frames=8). 32-frame eval
#    therefore feeds 24 untrained per-frame group tags. The comparison still
#    answers "does denser visual coverage help at all", but it is NOT a clean
#    result -- for that, retrain with data.num_frames: 32.
#  * K=128 at 32 frames -> S = 32*128 = 4096 visual tokens, predictor
#    self-attn ~12x the 8-frame cost. K128_BS defaults to 1. Bump only if it
#    fits. K=1 at 32 frames -> S = 32, trivial.
#
# Knobs:
#   REDO_8F=1   also re-run the 8-frame manifest into vsibench_8f/ for a clean
#               2x2 table (default 0: reuse the existing vsibench/ numbers).
#   K1_BS / K128_BS   eval batch size overrides.

set -euo pipefail

REDO_8F=${REDO_8F:-0}
K1_BS=${K1_BS:-8}
K128_BS=${K128_BS:-1}

ROOT="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_from_vljepa_ckpt_fusiongv"
CFG_K1="fusion_gv/configs/sft_from_vljepa_ckpt_fusiongv.yaml"
CFG_K128="fusion_gv/configs/sft_from_vljepa_ckpt_fusiongv_k128.yaml"
MANIFEST_32F="./data/vsibench_manifest_32f.jsonl"
MANIFEST_8F="./data/vsibench_manifest.jsonl"

mkdir -p "$LOG_DIR"

SIF="srun singularity exec \
    --nv \
    --env-file $ROOT/.env/hf.env \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -H $ROOT:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

run_eval () {  # $1=config  $2=manifest  $3=output_dir  $4=batch_size
    echo "[infer] cfg=$1 manifest=$2 -> $3  (bs=$4)  $(date)"
    $SIF python3 fusion_gv/infer_vsibench.py \
        --config "$1" \
        --config-section vsi_inference \
        --manifest "$2" \
        --output-dir "$3" \
        --batch-size "$4" \
        --no-grounding
    echo "[infer] done $(date)"
}

# ── 32-frame eval, both checkpoints ──────────────────────────────────────────
run_eval "$CFG_K1"   "$MANIFEST_32F" "./outputs/inference/sft_from_vljepa_ckpt_fusiongv/vsibench_32f"       "$K1_BS"
run_eval "$CFG_K128" "$MANIFEST_32F" "./outputs/inference/sft_from_vljepa_ckpt_fusiongv_k128/vsibench_32f"  "$K128_BS"

# ── optional: 8-frame re-run for a clean 2x2 ─────────────────────────────────
if [ "$REDO_8F" = "1" ]; then
    run_eval "$CFG_K1"   "$MANIFEST_8F" "./outputs/inference/sft_from_vljepa_ckpt_fusiongv/vsibench_8f"      "$K1_BS"
    run_eval "$CFG_K128" "$MANIFEST_8F" "./outputs/inference/sft_from_vljepa_ckpt_fusiongv_k128/vsibench_8f" 8
fi

# ── side-by-side diff ───────────────────────────────────────────────────────
echo "[compare] $(date)"
$SIF python3 fusion_gv/k_sweep/compare_vsibench.py \
    "K1-8f=./outputs/inference/sft_from_vljepa_ckpt_fusiongv/vsibench/vsibench_metrics.json" \
    "K128-8f=./outputs/inference/sft_from_vljepa_ckpt_fusiongv_k128/vsibench/vsibench_metrics.json" \
    "K1-32f=./outputs/inference/sft_from_vljepa_ckpt_fusiongv/vsibench_32f/vsibench_metrics.json" \
    "K128-32f=./outputs/inference/sft_from_vljepa_ckpt_fusiongv_k128/vsibench_32f/vsibench_metrics.json" \
    --baseline K1-8f \
    --out ./outputs/inference/vsibench_kcompare_32f.md

echo "All done at $(date)"
