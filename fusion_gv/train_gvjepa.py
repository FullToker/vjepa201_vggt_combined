#!/usr/bin/env python3
"""Entry point for FusionGVJEPA training.

Usage:
    python fusion_gv/train_gvjepa.py --config fusion_gv/configs/sft_vv_concat.yaml
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import yaml

from fusion_gv.gvjepa_trainer import (
    GVJEPATrainer,
    build_loader_from_config,
    build_model_from_config,
    build_optimizer_and_scheduler_from_config,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FusionGVJEPA")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["train"]
    _set_seed(tcfg.get("seed", 42))

    device_str = tcfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model     = build_model_from_config(cfg)
    loader    = build_loader_from_config(cfg)
    optimizer, scheduler = build_optimizer_and_scheduler_from_config(model, cfg)

    trainer = GVJEPATrainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        output_dir=tcfg["output_dir"],
        max_steps=tcfg["max_steps"],
        scheduler=scheduler,
        grad_accum_steps=tcfg.get("gradient_accumulation_steps", 1),
        clip_grad_norm=tcfg.get("clip_grad_norm", 1.0),
        temperature=tcfg.get("temperature", 0.07),
        log_every=tcfg.get("log_every", 20),
        save_every=tcfg.get("save_every", 1000),
        precision=tcfg.get("precision", "bf16"),
    )
    trainer.fit(device)


if __name__ == "__main__":
    main()
