"""
Spatial-alignment fusion (alternative to cross-attention).

JEPA tokens (24×24 = 576) are bilinearly interpolated up to VGGT's
37×37 = 1369 grid, then both streams are projected and added element-wise.

Compared with fusion.py (CrossAttention):
    ✓  O(N) per level instead of O(N_geo × N_sem)
    ✓  No encoder input changes — both keep native sizes (VGGT@518, JEPA@384)
    ✗  No cross-token interaction; each spatial position fused independently

Architecture (per level):
    jepa_up = bilinear(jepa_feat, 24×24 → 37×37)     (B*S, 1369, 1024)
    g  = geo_proj(vggt_feat) + carry_proj(carry)      (B*S, 1369, D_f)
    s  = sem_proj(jepa_up)                            (B*S, 1369, D_f)
    x     = LN_1( g + s )                            element-wise add
    fused = x + FFN( LN_2(x) )                       (B*S, 1369, D_f)

Level alignment:
    level 0  →  VGGT round  4  /  JEPA block  5   (early)
    level 1  →  VGGT round 11  /  JEPA block 11   (mid-early)
    level 2  →  VGGT round 17  /  JEPA block 17   (mid-late)
    level 3  →  VGGT round 23  /  JEPA block 23   (final)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FFN(nn.Module):
    def __init__(self, d: int, ratio: float, dropout: float):
        super().__init__()
        hidden = int(d * ratio)
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialAlign(nn.Module):
    """
    Bilinear upsample JEPA tokens from src_grid × src_grid to tgt_grid × tgt_grid.

    Args:
        src_grid : input  spatial grid size  (default 24, i.e. 24×24 = 576 tokens)
        tgt_grid : output spatial grid size  (default 37, i.e. 37×37 = 1369 tokens)
    """

    def __init__(self, src_grid: int = 24, tgt_grid: int = 37):
        super().__init__()
        self.src_grid = src_grid
        self.tgt_grid = tgt_grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B*S, src_grid², D)
        →   (B*S, tgt_grid², D)
        """
        BS, N, D = x.shape
        assert N == self.src_grid ** 2, (
            f"Expected {self.src_grid**2} tokens, got {N}"
        )
        # reshape to 2-D feature map, interpolate, flatten back
        x = x.view(BS, self.src_grid, self.src_grid, D).permute(0, 3, 1, 2)
        # (BS, D, src, src)
        x = F.interpolate(x, size=(self.tgt_grid, self.tgt_grid),
                          mode="bilinear", align_corners=False)
        # (BS, D, tgt, tgt)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        # (BS, tgt², D)
        return x


class AlignedFusionLevel(nn.Module):
    """
    One fusion level using element-wise addition after spatial alignment.

    Args:
        vggt_dim  : VGGT feature dim  (default 2048)
        jepa_dim  : JEPA feature dim  (default 1024)
        d_fusion  : common output dim (D_f)
        ffn_ratio : FFN hidden = d_fusion × ffn_ratio
        dropout   : dropout probability
        has_carry : inject carry from previous level into geometric stream
        src_grid  : JEPA spatial grid before alignment (default 24)
        tgt_grid  : target spatial grid = VGGT grid (default 37)
    """

    def __init__(
        self,
        vggt_dim: int = 2048,
        jepa_dim: int = 1024,
        d_fusion: int = 512,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        has_carry: bool = False,
        src_grid: int = 24,
        tgt_grid: int = 37,
    ):
        super().__init__()

        self.align    = SpatialAlign(src_grid=src_grid, tgt_grid=tgt_grid)
        self.geo_proj = nn.Linear(vggt_dim, d_fusion)
        self.sem_proj = nn.Linear(jepa_dim, d_fusion)
        self.carry_proj = nn.Linear(d_fusion, d_fusion) if has_carry else None

        self.norm1 = nn.LayerNorm(d_fusion)
        self.norm2 = nn.LayerNorm(d_fusion)
        self.ffn   = _FFN(d_fusion, ffn_ratio, dropout)

    def forward(
        self,
        vggt_feat: torch.Tensor,
        jepa_feat: torch.Tensor,
        carry: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        vggt_feat : (B*S, 1369, 2048)
        jepa_feat : (B*S,  576, 1024)
        carry     : (B*S, 1369, D_f) or None
        returns   : (B*S, 1369, D_f)
        """
        jepa_up = self.align(jepa_feat)          # (B*S, 1369, 1024)  24×24 → 37×37

        g = self.geo_proj(vggt_feat)             # (B*S, 1369, D_f)
        s = self.sem_proj(jepa_up)               # (B*S, 1369, D_f)

        if carry is not None and self.carry_proj is not None:
            g = g + self.carry_proj(carry)

        x     = self.norm1(g + s)                # element-wise add + LN
        fused = x + self.ffn(self.norm2(x))
        return fused


class AlignedMultiLevelFusion(nn.Module):
    """
    4-level spatial-alignment fusion with carry chain.

    Drop-in replacement for MultiLevelFusion in fusion.py.
    Same input/output contract:
        vggt_feats : list of 4 × (B, S, 1369, 2048)
        jepa_feats : list of 4 × (B, S,  576, 1024)
        returns    : list of 4 × (B, S, 1369, D_f)

    Args:
        num_levels : number of fusion levels (default 4)
        vggt_dim   : VGGT feature dim        (default 2048)
        jepa_dim   : JEPA feature dim        (default 1024)
        d_fusion   : output dim per patch    (default 512)
        ffn_ratio  : FFN hidden ratio        (default 4.0)
        dropout    : dropout probability     (default 0.0)
        src_grid   : JEPA grid size          (default 24)
        tgt_grid   : VGGT grid size          (default 37)
    """

    def __init__(
        self,
        num_levels: int = 4,
        vggt_dim: int = 2048,
        jepa_dim: int = 1024,
        d_fusion: int = 512,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        src_grid: int = 24,
        tgt_grid: int = 37,
    ):
        super().__init__()

        self.levels = nn.ModuleList([
            AlignedFusionLevel(
                vggt_dim=vggt_dim,
                jepa_dim=jepa_dim,
                d_fusion=d_fusion,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                has_carry=(i > 0),
                src_grid=src_grid,
                tgt_grid=tgt_grid,
            )
            for i in range(num_levels)
        ])

    def forward(
        self,
        vggt_feats: list[torch.Tensor],
        jepa_feats: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        vggt_feats : list of 4 × (B, S, 1369, 2048)
        jepa_feats : list of 4 × (B, S,  576, 1024)
        returns    : list of 4 × (B, S, 1369, D_f)
        """
        B, S = vggt_feats[0].shape[:2]

        carry = None
        fused_list = []

        for i, level in enumerate(self.levels):
            vg = vggt_feats[i].flatten(0, 1)     # (B*S, 1369, 2048)
            je = jepa_feats[i].flatten(0, 1)     # (B*S,  576, 1024)

            fused = level(vg, je, carry)          # (B*S, 1369, D_f)
            carry = fused

            fused_list.append(
                fused.view(B, S, fused.shape[-2], fused.shape[-1])
            )                                     # (B, S, 1369, D_f)

        return fused_list
