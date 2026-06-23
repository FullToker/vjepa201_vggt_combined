#!/bin/bash
#SBATCH --job-name=mlflow_server
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --partition=geo
#SBATCH --output=/glob/g01-cache/pf/Yushuo/logs/mlflow_server_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/logs/mlflow_server_%j.err

set -euo pipefail

mkdir -p /glob/g01-cache/pf/Yushuo/logs
echo "Node: $(hostname)  Started: $(date)"

srun singularity exec \
    -H /glob/g01-cache:/glob/g01-cache \
    /glob/g01-cache/container/nvpytorch24.05-py3.simg \
    bash -c "
        pip install --quiet --target=/glob/g01-cache/pf/Yushuo/pylibs mlflow && \
        PYTHONPATH=/glob/g01-cache/pf/Yushuo/pylibs \
        python -m mlflow server \
            --backend-store-uri file:///glob/g01-cache/pf/Yushuo/mlruns \
            --host 0.0.0.0 \
            --port 5000
    "
