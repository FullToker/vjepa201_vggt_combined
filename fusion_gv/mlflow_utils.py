"""Minimal MLflow helpers for FusionGVJEPA."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict


def load_env_file(path: str | Path | None) -> None:
    """Load simple KEY=VALUE or export KEY=VALUE lines into os.environ."""
    if path in {None, ""}:
        return
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MLflow env file not found: {path}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
        else:
            out[name] = str(value)
    return out


class NullMLflowLogger(AbstractContextManager):
    enabled = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def log_params(self, params: Dict[str, Any]) -> None:
        del params

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        del metrics, step

    def log_artifact(self, path: str | Path) -> None:
        del path

    def log_artifacts(self, paths) -> None:
        for path in paths:
            self.log_artifact(path)


class MLflowLogger(AbstractContextManager):
    enabled = True

    def __init__(self, mlflow):
        self.mlflow = mlflow

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
        return False

    def log_params(self, params: Dict[str, Any]) -> None:
        clean = {}
        for key, value in params.items():
            if any(word in key.lower() for word in ("password", "token", "secret")):
                continue
            clean[key] = str(value)[:500]
        if clean:
            self.mlflow.log_params(clean)

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if clean:
            self.mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            self.mlflow.log_artifact(str(path))

    def log_artifacts(self, paths) -> None:
        for path in paths:
            self.log_artifact(path)


def start_mlflow_run(cfg: Dict[str, Any]) -> AbstractContextManager:
    mcfg = cfg.get("mlflow", {})
    if not mcfg.get("enabled", False):
        return NullMLflowLogger()

    load_env_file(mcfg.get("env_file", ".env/mlflow.env"))

    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("mlflow is required when mlflow.enabled=true") from exc

    tracking_uri = mcfg.get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = mcfg.get("experiment_name") or os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    run_name = mcfg.get("run_name")
    if not run_name:
        xenc = cfg.get("fusion", {}).get("x_encoder_type", "xencoder")
        out_name = Path(cfg.get("train", {}).get("output_dir", "run")).name
        run_name = f"{xenc}-{out_name}"

    tags = dict(mcfg.get("tags", {}))
    tags.setdefault("x_encoder_type", cfg.get("fusion", {}).get("x_encoder_type", "unknown"))
    tags.setdefault("fusion_type", cfg.get("fusion", {}).get("fusion_type", "unknown"))

    mlflow.start_run(run_name=run_name, tags=tags)
    logger = MLflowLogger(mlflow)
    if mcfg.get("log_params", True):
        logger.log_params(flatten_dict(cfg))
    return logger
