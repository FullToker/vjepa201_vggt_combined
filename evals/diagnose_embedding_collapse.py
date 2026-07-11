#!/usr/bin/env python3
"""Diagnose embedding collapse in a FusionGVJEPA checkpoint.

Both VSI-Bench and MMSI-Bench candidate-matching accuracy are stuck at
~chance level (25% on 4-choice MC) across checkpoints. Checkpoint loading is
strict=True (fusion_gv/infer_gvjepa.py:92), so a shape/key mismatch would
have crashed rather than silently no-op'd -- ruling out a bad-load. This
script checks the next most likely cause: embedding collapse, i.e.
pred_embeddings / encode_target output vectors that barely vary across
different inputs, making cosine-candidate ranking effectively noise.

Randomly samples manifest rows (uniform over the whole file, not just the
head -- manifests are sorted by num_images, so a head-only sample would be
biased toward one image-count bucket), runs them through the model, and
reports:
  1. off-diagonal cosine-sim stats for pred_embeddings across different
     (images, query) samples
  2. off-diagonal cosine-sim stats for encode_target across distinct
     candidate option texts
  3. target-rank distribution: for each row, rank its own target string
     among its candidates by cosine-sim to that row's pred_embedding. This
     is the direct test of whether pred/target space alignment tracks
     correctness -- (1)/(2) can both look "healthy" (spread out, not
     collapsed) while still being misaligned, since collapse and alignment
     are separate failure modes.

Mean off-diagonal similarity close to 1.0 in (1)/(2) => collapsed (bad).
Mean target rank close to the random-chance expectation ((num_candidates+1)/2,
e.g. 2.5 for 4-choice MC) in (3) => pred/target spaces aren't aligned by
correctness even though embeddings vary -- points at a train/eval mismatch
(e.g. training loss uses a different pred/target pairing or head than
infer_gvjepa.py's eval-time scoring path) rather than at collapse.

Usage:
  python3 evals/diagnose_embedding_collapse.py \\
      --config fusion_gv/configs/infer_mmsibench.yaml \\
      --manifest ./data/mmsibench_manifest.jsonl \\
      --num-samples 24
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
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


def _index_offsets(manifest_path: Path) -> List[int]:
    offsets = []
    with open(manifest_path, "rb") as f:
        offset = f.tell()
        for raw in f:
            if raw.strip():
                offsets.append(offset)
            offset = f.tell()
    return offsets


def _sample_rows(manifest_path: Path, num_samples: int, seed: int) -> List[Dict[str, Any]]:
    offsets = _index_offsets(manifest_path)
    if not offsets:
        raise ValueError(f"Empty manifest: {manifest_path}")

    rng = random.Random(seed)
    k = min(num_samples, len(offsets))
    chosen = sorted(rng.sample(range(len(offsets)), k))

    rows = []
    with open(manifest_path, "rb") as f:
        for idx in chosen:
            f.seek(offsets[idx])
            rows.append(json.loads(f.readline()))
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
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
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

    rows = _sample_rows(Path(args.manifest), args.num_samples, args.seed)
    if len(rows) < 2:
        raise SystemExit("Need at least 2 manifest rows to compare embeddings.")
    print(f"Sampled {len(rows)} rows (seed={args.seed}) from {args.manifest}")

    # ---- 1. pred_embeddings across different samples (query+images) ----
    # MMSI-Bench rows have variable image count (2-10) -- unlike VSI-Bench's
    # fixed 8, a random sample mixes different S per row, so they can't be
    # torch.cat'd into one batched forward (preprocess() output shapes
    # differ in the S dim). Run one row at a time instead: a single-row
    # preprocess() output is already exactly what model._run_predictor
    # expects for batch=1 (images_vggt (1,S,3,518,518), images_jepa
    # (S,3,1,384,384)), so no reshaping needed, only the final pred_proj
    # outputs (uniform shape regardless of S) get concatenated.
    pred_embeddings_list = []
    for row in rows:
        imgs_v, imgs_j = preprocess(row["images"], need_vggt=need_vggt, need_jepa=True)
        with torch.no_grad():
            images_vggt = imgs_v.to(device) if imgs_v is not None else None
            images_jepa = imgs_j.to(device)
            _, pooled, _ = model._run_predictor(images_vggt, images_jepa, [row.get("query", "")])
            pred_embeddings_list.append(model.pred_proj(pooled))

    pred_embeddings = torch.cat(pred_embeddings_list, dim=0)
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

    # ---- 2b/3. within-question candidate separability + target rank ----
    # The global 35-candidate matrix above mixes two different effects: (a)
    # whether the 4 options of the SAME question are separable (the thing
    # that actually determines ranking correctness) and (b) baseline
    # similarity between DIFFERENT questions' option text, which is expected
    # to run higher than generic text since MC options here are all short,
    # same-domain spatial-relation phrases -- that's normal clustering, not
    # collapse. Restricting to each row's own candidates isolates (a).
    within_row_sims: List[float] = []
    ranks: List[int] = []
    n_candidates_seen: List[int] = []
    for i, row in enumerate(rows):
        candidates = row.get("candidates")
        target = row.get("target")
        if not candidates or len(candidates) < 2:
            continue
        with torch.no_grad():
            cand_emb = model.encode_target(candidates, device)
        cand_norm = F.normalize(cand_emb.float(), dim=-1)

        sim_within = cand_norm @ cand_norm.T
        n = sim_within.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=sim_within.device)
        within_row_sims.append(sim_within[mask].mean().item())

        if target in candidates:
            sims = (cand_norm @ pred_norm[i]).tolist()
            order = sorted(range(len(candidates)), key=lambda j: sims[j], reverse=True)
            rank = order.index(candidates.index(target)) + 1  # 1-indexed
            ranks.append(rank)
            n_candidates_seen.append(len(candidates))

    print("\n=== within-question candidate separability (per-row A/B/C/D vs each other, same question only) ===")
    if within_row_sims:
        m = statistics.mean(within_row_sims)
        s = statistics.stdev(within_row_sims) if len(within_row_sims) > 1 else 0.0
        hist_sorted = sorted(within_row_sims)
        print(f"rows checked: {len(within_row_sims)}")
        print(f"mean of per-row average off-diagonal sim: {m:.4f} (std across rows={s:.4f})")
        print(f"lowest/highest row: {hist_sorted[0]:.4f} / {hist_sorted[-1]:.4f}")
        print("  -> high value here (unlike the mixed 35-candidate test above) means the SAME")
        print("     question's own options look near-identical -- this is the collapse that")
        print("     would actually explain chance-level MC ranking, isolated from cross-question")
        print("     domain clustering.")
    else:
        print("No rows with >=2 candidates to check.")

    print("\n=== target rank among its own row's candidates ===")
    if ranks:
        mean_rank = statistics.mean(ranks)
        median_rank = statistics.median(ranks)
        top1_rate = sum(1 for r in ranks if r == 1) / len(ranks)
        avg_n = statistics.mean(n_candidates_seen)
        expected_random_rank = (avg_n + 1) / 2
        hist = dict(sorted(Counter(ranks).items()))
        print(f"rows scored: {len(ranks)}")
        print(f"mean rank: {mean_rank:.3f}  (random-chance expected: ~{expected_random_rank:.2f} for avg {avg_n:.1f} candidates)")
        print(f"median rank: {median_rank}")
        print(f"top1 hit rate: {top1_rate:.3f}")
        print(f"rank histogram (rank -> count): {hist}")
        print("  -> mean rank near the random-chance value means pred/target spaces aren't")
        print("     aligned by correctness, even if neither one individually looks collapsed.")
    else:
        print("No rows with valid candidates+target to compute rank.")

    print("\nDone.")


if __name__ == "__main__":
    main()
