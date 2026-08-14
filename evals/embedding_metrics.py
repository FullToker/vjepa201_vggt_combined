#!/usr/bin/env python3
"""Embedding-space eval metrics for FusionGVJEPA, decoupled from multiple-choice ranking.

infer_gvjepa.py --mode select only tests whether the correct candidate outranks
2-3 distractors drawn from the same question -- a tiny, easily-saturated-or-tanked
candidate pool that also requires SPAR-style {"candidates": [...]} rows. This script
instead scores the (pred, target) embedding space directly against any manifest of
plain {"images": [...], "query": "...", "target": "..."} rows (works on sentence/fill
manifests too, not just select-style MC ones -- see dataset/convert_spar_sftqa.py),
computing:

  1. held-out InfoNCE loss  -- same objective as training (gvjepa_trainer.py
     bidirectional_infonce), evaluated with the whole sampled set as one big batch
     of in-batch negatives. The most apples-to-apples continuous quality signal,
     but note it is *not* magnitude-comparable to a per-minibatch training-loss
     value unless --num-samples matches the training batch size (more negatives
     here makes it a harder, more informative number, not a worse one).
  2. alignment & uniformity -- Wang & Isola 2020 (arxiv 2005.10242) embedding-space
     diagnostics on unit-normalized embeddings, no downstream task needed:
       alignment  = mean_i ||pred_i - target_i||^2                (lower = better)
       uniformity = log mean_{i<j} exp(-2 ||z_i - z_j||^2)         (lower = more spread)
     Computed separately for pred and target embedding sets, so a collapsed side
     shows up directly (uniformity near 0 = that side's embeddings barely vary).
  3. positive/negative AUC -- ROC-AUC treating each row's own (pred, target) cosine
     similarity as the positive class and (pred, every other row's target) as
     negatives. Continuous alternative to top-1 MC accuracy; 0.5 = chance, 1.0 = every
     positive pair outranks every negative pair.
  4. per-sample margin -- cos(pred_i, target_i) - mean_j!=i cos(pred_i, target_j),
     the full per-row distribution (not a single accuracy scalar), so it separates
     "some samples fail hard" from "everything is uniformly mediocre".

Usage:
  python3 evals/embedding_metrics.py \\
      --config fusion_gv/configs/infer_spar.yaml \\
      --manifest ./data/spar_sftqa_eval_old.jsonl \\
      --num-samples 512
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from evals.embedding_metrics_lib import compute_embedding_metrics
from fusion_gv.gvjepa_trainer import build_model_from_config
from fusion_gv.infer_gvjepa import (
    _candidates,
    _infer_collate,
    _InferenceDataset,
    _load_checkpoint,
    _read_jsonl,
    _resolve_checkpoint,
    _target,
)


def _sample_rows_with_targets(manifest_path: Path, num_samples: int, seed: int) -> List[Dict[str, Any]]:
    rows = _read_jsonl(manifest_path)
    kept = []
    for row in rows:
        target_text, _ = _target(row, _candidates(row))
        if target_text:
            row["_target_text"] = target_text
            kept.append(row)
    if not kept:
        raise ValueError(f"No rows with a resolvable target text found in: {manifest_path}")

    rng = random.Random(seed)
    if len(kept) > num_samples:
        kept = rng.sample(kept, num_samples)
    return kept


@torch.no_grad()
def _encode_targets(model, texts: List[str], device: torch.device, batch_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(model.encode_target(texts[start : start + batch_size], device))
    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding-space eval metrics for FusionGVJEPA")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.07, help="Must match training temperature to be comparable.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Optional metrics JSON output path.")
    parser.add_argument(
        "--no-per-sample",
        action="store_true",
        help="Omit per_sample_margin from the printed/saved output (still computed).",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    icfg = cfg.get("inference", {})

    device_str = args.device or icfg.get("device") or cfg.get("train", {}).get("device", "cuda")
    if device_str != "cpu" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint)
    model = build_model_from_config(cfg)
    step = _load_checkpoint(model, checkpoint_path)
    model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint_path} (step={step})")

    x_encoder_type = cfg.get("fusion", {}).get("x_encoder_type", "fusion_gv")
    need_vggt = x_encoder_type == "fusion_gv"

    rows = _sample_rows_with_targets(Path(args.manifest), args.num_samples, args.seed)
    print(f"Sampled {len(rows)} rows (seed={args.seed}) with resolvable targets from {args.manifest}")

    input_root = icfg.get("input_root")
    dataset = _InferenceDataset(rows, input_root, need_vggt=need_vggt)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_infer_collate,
    )

    pred_chunks = []
    target_texts: List[str] = []
    precision = icfg.get("precision") or cfg.get("train", {}).get("precision", "bf16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    autocast_enabled = dtype is not None and device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            images_vggt = batch["images_vggt"].to(device)
            images_jepa = batch["images_jepa"].to(device)
            queries = batch["queries"]
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                pred = model.predict_embedding(images_vggt, images_jepa, queries)
            pred_chunks.append(pred.float().cpu())
            target_texts.extend(row["_target_text"] for row in batch["rows"])

    pred_embeddings = torch.cat(pred_chunks, dim=0).to(device)
    target_embeddings = _encode_targets(model, target_texts, device, args.target_batch_size).float()

    metrics = compute_embedding_metrics(pred_embeddings, target_embeddings, args.temperature)
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "step": step,
            "manifest": str(args.manifest),
            "temperature": args.temperature,
        }
    )

    per_sample = metrics.pop("per_sample_margin")
    if not args.no_per_sample:
        metrics["per_sample_margin"] = per_sample

    print("\n=== embedding_metrics ===")
    for key in (
        "num_samples",
        "held_out_infonce_loss",
        "alignment",
        "uniformity_pred",
        "uniformity_target",
        "auc",
        "margin_mean",
        "margin_std",
        "margin_min",
        "margin_max",
        "margin_frac_negative",
    ):
        print(f"{key}: {metrics[key]:.6f}" if isinstance(metrics[key], float) else f"{key}: {metrics[key]}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nwrote metrics: {output_path}")


if __name__ == "__main__":
    main()
