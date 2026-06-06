#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash train/train_vljepa_vv_fusion.sh <fusion_gv_config.yaml> <vjepa_config.yaml>

Runs two training experiments sequentially:
  1. VGGT + V-JEPA concat fusion  (fusion.x_encoder_type=fusion_gv)
  2. V-JEPA-only X-encoder        (fusion.x_encoder_type=vjepa)

Environment overrides:
  PYTHON_BIN=python3       Python executable used to run training
  DRY_RUN=1                Validate configs and print commands without training
  SKIP_VALIDATE=1          Skip config/path validation
USAGE
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

FUSION_CONFIG="$1"
VJEPA_CONFIG="$2"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

if [[ ! -f "$FUSION_CONFIG" ]]; then
  echo "[ERROR] fusion_gv config not found: $FUSION_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$VJEPA_CONFIG" ]]; then
  echo "[ERROR] vjepa config not found: $VJEPA_CONFIG" >&2
  exit 1
fi

validate_configs() {
  "$PYTHON_BIN" - "$FUSION_CONFIG" "$VJEPA_CONFIG" <<'INNER_PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

checks = [
    (Path(sys.argv[1]), "fusion_gv"),
    (Path(sys.argv[2]), "vjepa"),
]

seen_output_dirs = set()
errors = []
warnings = []


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def check_path(path_value, label, required=True):
    if path_value in (None, ""):
        if required:
            errors.append(f"missing {label}")
        return
    path = Path(str(path_value))
    if not path.exists():
        msg = f"{label} not found from cwd={Path.cwd()}: {path}"
        if required:
            errors.append(msg)
        else:
            warnings.append(msg)


for cfg_path, expected_xenc in checks:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fusion = cfg.get("fusion", {})
    data = cfg.get("data", {})
    train = cfg.get("train", {})
    mlflow = cfg.get("mlflow", {})

    xenc = fusion.get("x_encoder_type")
    if xenc != expected_xenc:
        errors.append(
            f"{cfg_path}: fusion.x_encoder_type should be {expected_xenc!r}, got {xenc!r}"
        )

    if fusion.get("fusion_type", "concat") != "concat":
        warnings.append(f"{cfg_path}: fusion.fusion_type is not concat: {fusion.get('fusion_type')!r}")

    manifests = as_list(data.get("train_manifests"))
    if not manifests:
        errors.append(f"{cfg_path}: data.train_manifests is empty or missing")
    for i, manifest in enumerate(manifests):
        check_path(manifest, f"{cfg_path}: data.train_manifests[{i}]")

    num_frames = data.get("num_frames")
    if num_frames is None:
        warnings.append(f"{cfg_path}: data.num_frames is missing")
    elif int(num_frames) != 4:
        warnings.append(f"{cfg_path}: data.num_frames={num_frames}; SPAR rows are expected to use 4 images")

    output_dir = train.get("output_dir")
    if not output_dir:
        errors.append(f"{cfg_path}: train.output_dir is missing")
    elif output_dir in seen_output_dirs:
        errors.append(f"{cfg_path}: train.output_dir duplicates another experiment: {output_dir}")
    else:
        seen_output_dirs.add(output_dir)

    if train.get("save_every", 0) not in (0, None):
        warnings.append(f"{cfg_path}: train.save_every={train.get('save_every')}; final-only checkpoints expect 0")

    if not train.get("batch_size"):
        warnings.append(f"{cfg_path}: train.batch_size is missing")
    if not train.get("gradient_accumulation_steps"):
        warnings.append(f"{cfg_path}: train.gradient_accumulation_steps is missing")

    check_path(fusion.get("jepa_ckpt"), f"{cfg_path}: fusion.jepa_ckpt")
    if expected_xenc == "fusion_gv":
        check_path(fusion.get("vggt_ckpt"), f"{cfg_path}: fusion.vggt_ckpt")

    if mlflow.get("enabled", False):
        check_path(mlflow.get("env_file", ".env/mlflow.env"), f"{cfg_path}: mlflow.env_file")
        run_name = mlflow.get("run_name")
        if not run_name:
            warnings.append(f"{cfg_path}: mlflow.run_name is null; train_gvjepa.py will auto-name it")

    print(f"[OK] {cfg_path}")
    print(f"     x_encoder_type={xenc}")
    print(f"     train_manifests={manifests}")
    print(f"     num_frames={num_frames}")
    print(f"     output_dir={output_dir}")
    print(f"     batch_size={train.get('batch_size')}")
    print(f"     gradient_accumulation_steps={train.get('gradient_accumulation_steps')}")
    print(f"     save_every={train.get('save_every')}")
    print(f"     mlflow.enabled={mlflow.get('enabled', False)}")

if warnings:
    print("\n[WARN]")
    for item in warnings:
        print(f"  - {item}")

if errors:
    print("\n[ERROR]")
    for item in errors:
        print(f"  - {item}")
    sys.exit(1)
INNER_PY
}

if [[ "$SKIP_VALIDATE" != "1" ]]; then
  echo "[INFO] Validating configs from cwd=$(pwd)"
  validate_configs
fi

run_train() {
  local name="$1"
  local config="$2"
  echo "[INFO] Starting ${name}: ${config}"
  echo "[CMD] ${PYTHON_BIN} fusion_gv/train_gvjepa.py --config ${config}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$PYTHON_BIN" fusion_gv/train_gvjepa.py --config "$config"
}

run_train "fusion_gv" "$FUSION_CONFIG"
run_train "vjepa" "$VJEPA_CONFIG"

echo "[INFO] Both training runs finished."
