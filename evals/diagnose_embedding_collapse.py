#!/usr/bin/env python3
"""Diagnose embedding collapse in a FusionGVJEPA checkpoint.

Both VSI-Bench and MMSI-Bench candidate-matching accuracy are stuck at
~chance level (25% on 4-choice MC) across checkpoints. Checkpoint loading is
strict=True (fusion_gv/infer_gvjepa.py:92), so a shape/key mismatch would
have crashed rather than silently no-op'd -- ruling out a bad-load. This
script checks the next most likely cause: embedding collapse, i.e.
pred_embeddings / encode_target output vectors that barely vary across
different inputs, making cosine-candidate ranking effectively noise.

Loads a handful of manifest rows, runs them through the model, and reports
the off-diagonal cosine-similarity stats for:
  1. pred_embeddings across different (images, query) samples
  2. encode_target embeddings across distinct candidate option texts

Mean off-diagonal similarity close to 1.0 => collapsed (bad). Spread out
(e.g. mean well below ~0.9, decent std) => embeddings do vary; the fault is
likely elsewhere (training objective / loss never converged / data pairing).

Usage:
  python3 fusion_gv/diagnose_embedding_collapse.py \\
      --config fusion_gv/configs/infer_mmsibench.yaml \\
      --manifest ./data/mmsibench_manifest.jsonl \\
      --num-samples 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml

from fusion_gv.gvjepa_trainer import build_model_from_config
from fusion_gv.infer_gvjepa import _load_checkpoint, _resolve_checkpoint
from fusion_gv.preprocess import preprocess


def _load_rows(manifest_path: Path, num_samples: int) -> List[Dict[str, Any]]:
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= num_samples:
                break
    return rows


def _off_diagonal_stats(sim: torch.Tensor) -> Tuple[float, float, float, float]:
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    vals = sim[mask]
    return vals.mean().item(), vals.std().item(), vals.min().item(), vals.max().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or cfg.get("inference", {}).get("device") or cfg.get("train", {}).get("device", "cuda")
    if device_str != "cpu" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint)
    model = build_model_from_config(cfg)
    step = _load_checkpoint(model, checkpoint_path)
    model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint_path} (step={step})")

    need_vggt = cfg.get("fusion", {}).get("x_encoder_type", "fusion_gv") == "fusion_gv"

    rows = _load_rows(Path(args.manifest), args.num_samples)
    if len(rows) < 2:
        raise SystemExit("Need at least 2 manifest rows to compare embeddings.")
    print(f"Loaded {len(rows)} rows from {args.manifest}")

    # ---- 1. pred_embeddings across different samples (query+images) ----
    vggt_list, jepa_list, queries = [], [], []
    for row in rows:
        imgs_v, imgs_j = preprocess(row["images"], need_vggt=need_vggt, need_jepa=True)
        if imgs_v is not None:
            vggt_list.append(imgs_v)
        jepa_list.append(imgs_j)
        queries.append(row.get("query", ""))

    with torch.no_grad():
        images_vggt = torch.cat(vggt_list, dim=0).to(device) if vggt_list else None
        images_jepa = torch.cat([j.unsqueeze(0) for j in jepa_list], dim=0).flatten(0, 1).to(device)
        _, pooled, _ = model._run_predictor(images_vggt, images_jepa, queries)
        pred_embeddings = model.pred_proj(pooled)

    pred_norm = F.normalize(pred_embeddings.float(), dim=-1)
    sim_pred = pred_norm @ pred_norm.T
    mean, std, mn, mx = _off_diagonal_stats(sim_pred)
    print("\n=== pred_embeddings (across different rows: images+query) ===")
    print(f"off-diagonal cosine sim: mean={mean:.4f} std={std:.4f} min={mn:.4f} max={mx:.4f}")
    print("  -> mean close to 1.0 means predictions barely vary across different inputs (collapsed).")

    # ---- 2. candidate text embeddings (encode_target) across distinct options ----
    seen = set()
    uniq_candidates: List[str] = []
    for row in rows:
        for c in row.get("candidates") or []:
            if c not in seen:
                seen.add(c)
                uniq_candidates.append(c)
        if len(uniq_candidates) >= 32:
            break

    if len(uniq_candidates) >= 2:
        with torch.no_grad():
            target_emb = model.encode_target(uniq_candidates, device)
        target_norm = F.normalize(target_emb.float(), dim=-1)
        sim_target = target_norm @ target_norm.T
        mean, std, mn, mx = _off_diagonal_stats(sim_target)
        print(f"\n=== encode_target (across {len(uniq_candidates)} distinct candidate texts) ===")
        print(f"off-diagonal cosine sim: mean={mean:.4f} std={std:.4f} min={mn:.4f} max={mx:.4f}")
        print("  -> mean close to 1.0 means the text encoder isn't distinguishing between different options.")
    else:
        print("\nNo candidates found in manifest rows to test encode_target.")

    print("\nDone.")


if __name__ == "__main__":
    main()
