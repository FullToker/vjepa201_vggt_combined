"""
Single-level spatial-alignment fusion: per-stream LN+MLP projector, then align+concat.

Only the final encoder level is used (no multi-level fusion, no cross-attention).

Architecture:
    geo = GELU-MLP(LN(vggt_feat))            (B,S,1369, proj_dim)
    sem = GELU-MLP(LN(jepa_feat))            (B,S, 576, proj_dim)
    sem = bilinear(sem, 24×24 → 37×37)       (B,S,1369, proj_dim)
    fused = cat([geo, sem], dim=-1)          (B,S,1369, 2*proj_dim)

Output dim = 2 * proj_dim (default 2 * 1024 = 2048).
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


class SingleLevelFusion(nn.Module):
    """
    Final-level-only fusion: per-stream LayerNorm + 2-layer MLP projector,
    JEPA spatially aligned to VGGT's grid, then channel-concat.

    Args:
        vggt_dim : VGGT feature dim           (default 2048)
        jepa_dim : JEPA feature dim           (default 1024)
        proj_dim : per-stream projected dim   (default 1024) → fused dim = 2*proj_dim
        src_grid : JEPA spatial grid size     (default 24)
        tgt_grid : VGGT spatial grid size     (default 37)
    """

    def __init__(
        self,
        vggt_dim: int = 2048,
        jepa_dim: int = 1024,
        proj_dim: int = 1024,
        src_grid: int = 24,
        tgt_grid: int = 37,
        **kwargs,   # silently absorb unrelated config fields
    ):
        super().__init__()
        self.out_dim = proj_dim * 2

        self.geo_proj = nn.Sequential(
            nn.LayerNorm(vggt_dim),
            nn.Linear(vggt_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.sem_proj = nn.Sequential(
            nn.LayerNorm(jepa_dim),
            nn.Linear(jepa_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.align = SpatialAlign(src_grid=src_grid, tgt_grid=tgt_grid)

    def forward(
        self,
        vggt_feat: torch.Tensor,
        jepa_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        vggt_feat : (B, S, 1369, vggt_dim)   final level only
        jepa_feat : (B, S,  576, jepa_dim)   final level only
        returns   : (B, S, 1369, 2*proj_dim)
        """
        B, S = vggt_feat.shape[:2]

        geo = self.geo_proj(vggt_feat)                 # (B,S,1369,proj_dim)

        sem = self.sem_proj(jepa_feat)                  # (B,S, 576,proj_dim)
        sem = self.align(sem.flatten(0, 1))             # (B*S,1369,proj_dim)
        sem = sem.view(B, S, *sem.shape[1:])            # (B,S,1369,proj_dim)

        return torch.cat([geo, sem], dim=-1)             # (B,S,1369,2*proj_dim)
