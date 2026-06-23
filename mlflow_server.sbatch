#!/bin/bash
#SBATCH --job-name=mlflow_server
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/logs/mlflow_server_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/logs/mlflow_server_%j.err

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo"
CONTAINER="/glob/g01-cache/container/nvpytorch24.05-py3.simg"
MLRUNS="${ROOT}/mlruns"
PORT=5000

mkdir -p "${ROOT}/logs"

SIF="singularity exec \
    -H ${ROOT}:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    ${CONTAINER}"

# install mlflow to persistent home (~/.local) if not present
$SIF pip install --user --quiet mlflow

NODE=$(hostname)
echo "MLflow server starting on node: ${NODE}, port: ${PORT}"
echo "SSH tunnel: ssh -L ${PORT}:${NODE}:${PORT} <login-node>"
echo "UI: http://localhost:${PORT}"
echo "Started: $(date)"
echo "---"

srun $SIF mlflow server \
    --backend-store-uri "file://${MLRUNS}" \
    --host 0.0.0.0 \
    --port "${PORT}"
