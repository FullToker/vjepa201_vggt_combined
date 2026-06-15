"""
GroundingHead: multi-layer cross-attention grounding module.

Q  = predictor visual summary  (B*S, 1,    h)   — query-conditioned per-view token
KV = projected spatial feat    (B*S, 1369, h)   — FusionGV patch features before pooling

Each CrossAttnLayer refines Q across spatial KV.
Final layer returns raw (pre-softmax) logits (B*S, H, 1, 1369)
→ mean over heads → squeeze → reshape → (B*S, patch_grid, patch_grid) heatmap.
Train with sigmoid + BCE against bbox-derived binary patch mask.
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
        spatial (B,S,N,D_f) → spatial_proj → kv (B*S, N, h)
        summary (B,S,h)     → reshape      → q  (B*S, 1, h)
                                    ↓
                        CrossAttnLayer × num_layers
                        each layer: q attends kv, q refined
                                    ↓
                    last layer raw logits (B*S, 1, N)
                        → squeeze(1) → (B*S, N)
                        → reshape    → (B*S, G, G)

    G = patch_grid = img_size / patch_size = 518 / 14 = 37.

    Caller loss:
        F.binary_cross_entropy_with_logits(logits, gt_mask)
        gt_mask: (B*S, G, G) binary float, derived from projected 2D bbox.

    Args:
        spatial_dim:  D_f from FusionGV (3072 for concat fusion)
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
        summary: torch.Tensor,   # (B, S, h)
        spatial: torch.Tensor,   # (B, S, N_patches, D_f)
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B*S, patch_grid, patch_grid)  raw pre-softmax attention logits
        """
        B, S, N, D_f = spatial.shape

        q  = summary.reshape(B * S, 1, summary.shape[-1])
        kv = self.spatial_proj(spatial.reshape(B * S, N, D_f))  # (B*S, N, h)

        raw_logits = None
        for i, layer in enumerate(self.layers):
            q, raw_logits = layer(q, kv, return_raw_logits=(i == self.num_layers - 1))

        # raw_logits: (B*S, 1, N) → (B*S, G, G)
        return raw_logits.squeeze(1).reshape(B * S, self.patch_grid, self.patch_grid)
