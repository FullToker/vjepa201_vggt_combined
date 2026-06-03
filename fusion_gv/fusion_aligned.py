"""
Spatial-alignment fusion: concat without projection.

JEPA tokens (24×24 = 576) are bilinearly interpolated up to VGGT's
37×37 = 1369 grid, then both feature vectors are concatenated along
the channel dimension. No learnable parameters — zero information loss.

Architecture (per level):
    jepa_up = bilinear(jepa_feat, 24×24 → 37×37)    (B*S, 1369, 1024)
    fused   = cat([vggt_feat, jepa_up], dim=-1)      (B*S, 1369, 3072)

Output dim = vggt_dim + jepa_dim = 2048 + 1024 = 3072 (fixed).

Level alignment:
    level 0  →  VGGT round  4  /  JEPA block  5   (early)
    level 1  →  VGGT round 11  /  JEPA block 11   (mid-early)
    level 2  →  VGGT round 17  /  JEPA block 17   (mid-late)
    level 3  →  VGGT round 23  /  JEPA block 23   (final)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAlign(nn.Module):
    """
    Bilinear upsample JEPA tokens from src_grid × src_grid to tgt_grid × tgt_grid.
    No learnable parameters.

    Args:
        src_grid : input  spatial grid size  (default 24 → 576 tokens)
        tgt_grid : output spatial grid size  (default 37 → 1369 tokens)
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
        assert N == self.src_grid ** 2, f"Expected {self.src_grid**2} tokens, got {N}"
        x = x.view(BS, self.src_grid, self.src_grid, D).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(self.tgt_grid, self.tgt_grid),
                          mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        return x   # (B*S, tgt_grid², D)


class AlignedMultiLevelFusion(nn.Module):
    """
    4-level spatial-alignment fusion via direct concatenation.
    No learnable parameters in this module.

    Input / output contract (same interface as MultiLevelFusion):
        vggt_feats : list of N × (B, S, 1369, 2048)
        jepa_feats : list of N × (B, S,  576, 1024)
        returns    : list of N × (B, S, 1369, 3072)

    Args:
        num_levels : number of fusion levels    (default 4)
        vggt_dim   : VGGT feature dim           (default 2048)
        jepa_dim   : JEPA feature dim           (default 1024)
        src_grid   : JEPA spatial grid size     (default 24)
        tgt_grid   : VGGT spatial grid size     (default 37)
        **kwargs   : absorbs unused params (d_fusion, num_heads, …) from config
    """

    def __init__(
        self,
        num_levels: int = 4,
        vggt_dim: int = 2048,
        jepa_dim: int = 1024,
        src_grid: int = 24,
        tgt_grid: int = 37,
        **kwargs,                    # silently absorb d_fusion, ffn_ratio, etc.
    ):
        super().__init__()
        self.num_levels = num_levels
        self.out_dim    = vggt_dim + jepa_dim   # 3072
        self.align      = SpatialAlign(src_grid=src_grid, tgt_grid=tgt_grid)

    def forward(
        self,
        vggt_feats: list[torch.Tensor],
        jepa_feats: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        vggt_feats : list of 4 × (B, S, 1369, 2048)
        jepa_feats : list of 4 × (B, S,  576, 1024)
        returns    : list of 4 × (B, S, 1369, 3072)
        """
        B, S = vggt_feats[0].shape[:2]
        fused_list = []

        for i in range(self.num_levels):
            vg     = vggt_feats[i].flatten(0, 1)          # (B*S, 1369, 2048)
            je_up  = self.align(jepa_feats[i].flatten(0, 1))  # (B*S, 1369, 1024)
            fused  = torch.cat([vg, je_up], dim=-1)        # (B*S, 1369, 3072)
            fused_list.append(fused.view(B, S, fused.shape[-2], fused.shape[-1]))

        return fused_list   # 4 × (B, S, 1369, 3072)
