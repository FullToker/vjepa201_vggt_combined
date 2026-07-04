#!/usr/bin/env python3
"""Entry point for FusionGVJEPA training.

Usage:
    python fusion_gv/train_gvjepa.py --config fusion_gv/configs/sft_vv_concat.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs

from fusion_gv.mlflow_utils import start_mlflow_run
from fusion_gv.gvjepa_trainer import (
    GVJEPATrainer,
    build_loader_from_config,
    build_grounding_loader_from_config,
    build_model_from_config,
    build_optimizer_and_scheduler_from_config,
)


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

    S = collected[0]["images_vggt"].shape[1]
    fixed = {
        "images_vggt": torch.cat([b["images_vggt"] for b in collected])[:n_samples],
        "images_jepa": torch.cat([b["images_jepa"] for b in collected])[:n_samples * S],
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

    tcfg = cfg["train"]
    precision = tcfg.get("precision", "bf16")
    mixed_precision = precision if precision in {"bf16", "fp16"} else "no"
    # grounding vs infonce steps use disjoint parameter subsets each iteration
    # (see FusionGVJEPA.forward mode dispatch) — DDP needs find_unused_parameters
    # to tolerate the varying used-parameter set across steps.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=mixed_precision, kwargs_handlers=[ddp_kwargs])

    _set_seed(tcfg.get("seed", 42) + accelerator.process_index)

    model            = build_model_from_config(cfg)
    loader           = build_loader_from_config(cfg)
    grounding_loader = build_grounding_loader_from_config(cfg)
    optimizer, scheduler = build_optimizer_and_scheduler_from_config(model, cfg)

    # prepare: Accelerate handles device placement, DDP wrap, DistributedSampler
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if grounding_loader is not None:
        grounding_loader = accelerator.prepare(grounding_loader)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    if args.overfit:
        if accelerator.is_main_process:
            print(f"[overfit] Freezing {args.overfit_samples} samples, running {args.overfit_steps} steps...")
        loader = _build_overfit_loader(loader, args.overfit_samples)
        if grounding_loader is not None:
            grounding_loader = _build_overfit_loader(grounding_loader, args.overfit_samples)
        tcfg["max_steps"] = args.overfit_steps
        tcfg["log_every"] = 10
        tcfg["save_every"] = 0
        scheduler = None

    output_dir = Path(tcfg["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_copy = output_dir / "config.yaml"
        with open(config_copy, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        config_copy = None

    with start_mlflow_run(cfg if accelerator.is_main_process else {}) as mlflow_logger:
        if accelerator.is_main_process and getattr(mlflow_logger, "enabled", False) and config_copy:
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
            grounding_loss_weight=gcfg.get("loss_weight", 0.1),
            grounding_pos_weight=gcfg.get("pos_weight", 5.0),
            suppression_loss_weight=gcfg.get("suppression_loss_weight", 0.1),
            mlflow_logger=mlflow_logger,
            accelerator=accelerator,
        )
        trainer.fit()


if __name__ == "__main__":
    main()
