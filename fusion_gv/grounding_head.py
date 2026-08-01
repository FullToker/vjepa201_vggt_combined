"""
GroundingHead: multi-layer cross-attention grounding module.

Q  = predictor visual summary  (B*S, K,    h)   — K query-conditioned tokens per frame
                                                    (K=1 with today's mean-pooled predictor
                                                     input; K>1 if a future resampler keeps
                                                     multiple tokens per frame)
KV = projected spatial feat    (B*S, 1369, h)   — FusionGV patch features before pooling

Each CrossAttnLayer refines Q across spatial KV.
Final layer returns raw (pre-softmax) logits (B*S, H, K, 1369)
→ mean over heads → mean over K → reshape → (B*S, patch_grid, patch_grid) heatmap.
Averaging over K keeps the logit scale independent of K (sum would scale linearly
with K and push the loss into its saturated regime, and would require
re-tuning focal_alpha/focal_beta every time K changes).
Train with sigmoid + CenterNet-style modified focal loss against a bbox-derived
Gaussian heatmap (peak 1.0 at box center) — see gvjepa_trainer.modified_focal_loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnLayer(nn.Module):
    """Pre-norm cross-attention layer that optionally exposes raw logits."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        ffn_dim = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,                # (B, Nq,  d)
        kv: torch.Tensor,               # (B, Nkv, d)
        return_raw_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            out:    (B, Nq, d)           refined query tokens
            logits: (B, Nq, Nkv) | None  head-averaged raw pre-softmax logits
        """
        B, Nq,  d = q.shape
        Nkv       = kv.shape[1]
        H, Hd     = self.n_heads, self.head_dim

        # cross-attention with pre-norm on Q
        q_norm = self.norm1(q)
        Q = self.q_proj(q_norm).reshape(B, Nq,  H, Hd).transpose(1, 2)  # (B,H,Nq,Hd)
        K = self.k_proj(kv).reshape(B,   Nkv,  H, Hd).transpose(1, 2)  # (B,H,Nkv,Hd)
        V = self.v_proj(kv).reshape(B,   Nkv,  H, Hd).transpose(1, 2)  # (B,H,Nkv,Hd)

        raw  = (Q @ K.transpose(-1, -2)) * self.scale                   # (B,H,Nq,Nkv)
        attn = F.softmax(raw, dim=-1)
        ctx  = (attn @ V).transpose(1, 2).reshape(B, Nq, d)             # (B,Nq,d)
        out  = q + self.out_proj(ctx)

        # FFN with pre-norm
        out = out + self.ffn(self.norm2(out))

        # mean over heads: (B,H,Nq,Nkv) → (B,Nq,Nkv)
        logits = raw.mean(dim=1) if return_raw_logits else None
        return out, logits


class GroundingHead(nn.Module):
    """
    Grounding head: spatial skip-connect + multi-layer cross-attention.

    Flow:
        spatial (B,S,N,D_f)   → spatial_proj → kv (B*S, N, h)
        summary (B,S*K,h)     → reshape      → q  (B*S, K, h)
                                      ↓
                          CrossAttnLayer × num_layers
                          each layer: q attends kv, q refined
                                      ↓
                      last layer raw logits (B*S, K, N)
                          → mean(dim=1) over K → (B*S, N)
                          → reshape             → (B*S, G, G)

    G = patch_grid = img_size / patch_size = 518 / 14 = 37.

    Caller loss:
        gvjepa_trainer.modified_focal_loss(logits, gt_heatmap)
        gt_heatmap: (B*S, G, G) unnormalized Gaussian float (peak 1.0 at box
        center), derived from projected 2D bbox via boxes_to_gaussian_heatmap.

    Args:
        spatial_dim:  D_f from FusionGV (2048 by default, = 2 * proj_dim)
        hidden_dim:   h — must equal predictor_hidden_size
        num_layers:   cross-attn depth
        num_heads:    attention heads (must divide hidden_dim)
        patch_grid:   G, default 37
    """

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        patch_grid: int = 37,
    ) -> None:
        super().__init__()
        self.patch_grid = patch_grid
        self.num_layers = num_layers

        self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)
        self.layers = nn.ModuleList([
            CrossAttnLayer(hidden_dim, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        summary: torch.Tensor,   # (B, S*K, h) — K tokens per frame, K=1 today
        spatial: torch.Tensor,   # (B, S, N_patches, D_f)
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B*S, patch_grid, patch_grid)  raw pre-softmax attention logits,
                     averaged over the K tokens of each frame
        """
        B, S, N, D_f = spatial.shape
        assert summary.shape[1] % S == 0, (
            f"summary length {summary.shape[1]} must be a multiple of S={S}"
        )
        K = summary.shape[1] // S

        # summary must be laid out frame-major: [frame0_tok0..K-1, frame1_tok0..K-1, ...]
        q  = summary.reshape(B * S, K, summary.shape[-1])       # (B*S, K, h)
        kv = self.spatial_proj(spatial.reshape(B * S, N, D_f))  # (B*S, N, h)

        raw_logits = None
        for i, layer in enumerate(self.layers):
            q, raw_logits = layer(q, kv, return_raw_logits=(i == self.num_layers - 1))

        # raw_logits: (B*S, K, N) → mean over K (not sum: keeps logit scale K-invariant) → (B*S, G, G)
        raw_logits = raw_logits.mean(dim=1)
        return raw_logits.reshape(B * S, self.patch_grid, self.patch_grid)
