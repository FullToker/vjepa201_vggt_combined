#!/usr/bin/env python3
"""Shared embedding-space metrics: held-out InfoNCE, alignment/uniformity
(Wang & Isola 2020, arxiv 2005.10242), positive/negative AUC, per-sample margin.

Pure torch, no fusion_gv/openvljepa imports -- both evals/embedding_metrics.py
(FusionGVJEPA) and openvl/infer_bench.py (raw open-vljepa) call into this same
module so the two architectures' numbers are computed identically and stay
directly comparable. See evals/diagnose_embedding_collapse.py and
evals/embedding_metrics.py module docstrings for the reasoning behind each
metric.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def bidirectional_infonce(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Bidirectional InfoNCE between (B, D) pred and (B, D) target tensors.

    Same objective fusion_gv trains against (fusion_gv/gvjepa_trainer.py::
    bidirectional_infonce) -- duplicated here rather than imported so this
    module stays free of fusion_gv's (and open-vljepa's) dependencies.
    """
    if pred.shape[0] < 2:
        raise ValueError("InfoNCE requires batch size >= 2.")
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = (pred @ target.T) / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _pairwise_sq_dist(z_norm: torch.Tensor) -> torch.Tensor:
    """||z_i - z_j||^2 for unit-normalized z, via the cosine identity (no cdist)."""
    sim = z_norm @ z_norm.T
    return (2.0 - 2.0 * sim).clamp_min(0.0)


def _uniformity(z_norm: torch.Tensor, t: float = 2.0) -> float:
    n = z_norm.shape[0]
    sq_dist = _pairwise_sq_dist(z_norm)
    mask = ~torch.eye(n, dtype=torch.bool, device=z_norm.device)
    return torch.log(torch.exp(-t * sq_dist[mask]).mean()).item()


def _roc_auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """Mann-Whitney AUC = P(random positive score > random negative score)."""
    n_pos, n_neg = pos_scores.numel(), neg_scores.numel()
    all_scores = torch.cat([pos_scores, neg_scores]).double()
    order = torch.argsort(all_scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, all_scores.numel() + 1, dtype=torch.float64, device=all_scores.device)
    sum_rank_pos = ranks[:n_pos].sum()
    return ((sum_rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)).item()


def compute_embedding_metrics(
    pred_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> Dict[str, Any]:
    """Score a (pred_i, target_i) embedding set with the full corpus as
    in-batch negatives. Row i's positive is target_i; every other row's
    target is a negative for row i.

    Returns held_out_infonce_loss, alignment, uniformity_pred,
    uniformity_target, auc, margin_mean/std/min/max/frac_negative, and
    per_sample_margin (one value per input row, same order as the inputs).
    """
    n = pred_embeddings.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to compute corpus-level metrics.")

    pred_norm = F.normalize(pred_embeddings.float(), dim=-1)
    target_norm = F.normalize(target_embeddings.float(), dim=-1)

    infonce_loss = bidirectional_infonce(pred_embeddings.float(), target_embeddings.float(), temperature).item()

    alignment = (2.0 - 2.0 * (pred_norm * target_norm).sum(dim=-1)).mean().item()
    uniformity_pred = _uniformity(pred_norm)
    uniformity_target = _uniformity(target_norm)

    sim = pred_norm @ target_norm.T  # (n, n): sim[i, j] = cos(pred_i, target_j)
    pos_scores = torch.diagonal(sim)
    neg_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    neg_scores = sim[neg_mask]
    auc = _roc_auc(pos_scores, neg_scores)

    row_sum = sim.sum(dim=1)
    neg_mean_per_row = (row_sum - pos_scores) / (n - 1)
    margins = (pos_scores - neg_mean_per_row).tolist()
    margin_tensor = torch.tensor(margins)

    return {
        "num_samples": n,
        "held_out_infonce_loss": infonce_loss,
        "alignment": alignment,
        "uniformity_pred": uniformity_pred,
        "uniformity_target": uniformity_target,
        "auc": auc,
        "margin_mean": margin_tensor.mean().item(),
        "margin_std": margin_tensor.std().item(),
        "margin_min": margin_tensor.min().item(),
        "margin_max": margin_tensor.max().item(),
        "margin_frac_negative": (margin_tensor < 0).float().mean().item(),
        "per_sample_margin": margins,
    }
