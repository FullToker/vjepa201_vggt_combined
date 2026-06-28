#!/usr/bin/env python3
"""Entry point for FusionGVJEPA training.

Usage:
    python fusion_gv/train_gvjepa.py --config fusion_gv/configs/sft_vv_concat.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
import yaml

from fusion_gv.mlflow_utils import start_mlflow_run
from fusion_gv.gvjepa_trainer import (
    GVJEPATrainer,
    build_loader_from_config,
    build_grounding_loader_from_config,
    build_model_from_config,
    build_optimizer_and_scheduler_from_config,
)


def _init_distributed() -> tuple[int, int, int]:
    """Init NCCL if torchrun sets LOCAL_RANK, else single-GPU fallback.

    Returns (rank, local_rank, world_size).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_overfit_loader(loader, n_samples: int):
    """Freeze a single batch of n_samples and repeat it indefinitely."""
    collected, total = [], 0
    for batch in loader:
        collected.append(batch)
        total += batch["images_vggt"].shape[0]
        if total >= n_samples:
            break

    fixed = {
        "images_vggt": torch.cat([b["images_vggt"] for b in collected])[:n_samples],
        "images_jepa": torch.cat([b["images_jepa"] for b in collected])[:n_samples],
        "query":  sum([b["query"]  for b in collected], [])[:n_samples],
        "target": sum([b["target"] for b in collected], [])[:n_samples],
        "boxes":  sum([b["boxes"]  for b in collected], [])[:n_samples],
    }

    class _RepeatLoader:
        def __iter__(self):
            while True:
                yield fixed

    return _RepeatLoader()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FusionGVJEPA")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument(
        "--x-encoder-type",
        choices=("fusion_gv", "vjepa"),
        default=None,
        help="Override fusion.x_encoder_type from the YAML config",
    )
    parser.add_argument(
        "--x-encoder-output-dim",
        type=int,
        default=None,
        help="Override fusion.x_encoder_output_dim from the YAML config",
    )
    parser.add_argument(
        "--overfit", action="store_true",
        help="Freeze 128 samples and overfit for 200 steps to verify all components",
    )
    parser.add_argument(
        "--overfit-samples", type=int, default=128,
        help="Number of samples to freeze for overfit test (default: 128)",
    )
    parser.add_argument(
        "--overfit-steps", type=int, default=200,
        help="Steps to run in overfit test (default: 200)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("fusion", {})
    if args.x_encoder_type is not None:
        cfg["fusion"]["x_encoder_type"] = args.x_encoder_type
    if args.x_encoder_output_dim is not None:
        cfg["fusion"]["x_encoder_output_dim"] = args.x_encoder_output_dim

    rank, local_rank, world_size = _init_distributed()
    is_main = (rank == 0)

    tcfg = cfg["train"]
    _set_seed(tcfg.get("seed", 42) + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # build raw model first so parameter_groups() works before DDP wrapping
    model                = build_model_from_config(cfg)
    loader               = build_loader_from_config(cfg, rank=rank, world_size=world_size)
    grounding_loader     = build_grounding_loader_from_config(cfg, rank=rank, world_size=world_size)
    optimizer, scheduler = build_optimizer_and_scheduler_from_config(model, cfg)

    model.to(device)
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if args.overfit:
        if is_main:
            print(f"[overfit] Freezing {args.overfit_samples} samples, running {args.overfit_steps} steps...")
        loader = _build_overfit_loader(loader, args.overfit_samples)
        if grounding_loader is not None:
            grounding_loader = _build_overfit_loader(grounding_loader, args.overfit_samples)
        tcfg["max_steps"] = args.overfit_steps
        tcfg["log_every"] = 10
        tcfg["save_every"] = 0
        scheduler = None

    output_dir = Path(tcfg["output_dir"])
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_copy = output_dir / "config.yaml"
        with open(config_copy, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        config_copy = None

    # only rank-0 talks to MLflow; other ranks get a no-op logger
    with start_mlflow_run(cfg if is_main else {}) as mlflow_logger:
        if is_main and getattr(mlflow_logger, "enabled", False) and config_copy:
            mlflow_logger.log_artifact(config_copy)

        gcfg = cfg.get("grounding", {})
        trainer = GVJEPATrainer(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            output_dir=tcfg["output_dir"],
            max_steps=tcfg["max_steps"],
            scheduler=scheduler,
            grounding_loader=grounding_loader,
            grounding_ratio=gcfg.get("batch_ratio", 5),
            grad_accum_steps=1 if args.overfit else tcfg.get("gradient_accumulation_steps", 1),
            clip_grad_norm=tcfg.get("clip_grad_norm", 1.0),
            temperature=tcfg.get("temperature", 0.07),
            log_every=tcfg.get("log_every", 20),
            save_every=tcfg.get("save_every", 0),
            precision=tcfg.get("precision", "bf16"),
            grounding_loss_weight=gcfg.get("loss_weight", 0.1),
            grounding_pos_weight=gcfg.get("pos_weight", 5.0),
            suppression_loss_weight=gcfg.get("suppression_loss_weight", 0.1),
            mlflow_logger=mlflow_logger,
            rank=rank,
            world_size=world_size,
        )
        trainer.fit(device)


if __name__ == "__main__":
    main()
