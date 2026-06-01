"""
Multi-level cross-attention fusion module.

Architecture (per level):
    Q = geo_proj(vggt_feat) + carry_proj(carry)   [carry injected from previous level]
    K = sem_proj(jepa_feat)
    V = sem_proj(jepa_feat)
    x     = LN( Q + CrossAttn(Q, K, V) )
    fused = x + FFN( LN(x) )

Level alignment:
    level 0  →  VGGT round  4  /  JEPA block  5   (early)
    level 1  →  VGGT round 11  /  JEPA block 11   (mid-early)
    level 2  →  VGGT round 17  /  JEPA block 17   (mid-late)
    level 3  →  VGGT round 23  /  JEPA block 23   (final)
"""

import torch
import torch.nn as nn


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


class CrossAttnLevel(nn.Module):
    """
    One fusion level.

    Args:
        vggt_dim  : input dim from VGGT  (2048)
        jepa_dim  : input dim from JEPA  (1024)
        d_fusion  : common attention dim (D_f)
        num_heads : attention heads
        ffn_ratio : FFN hidden = d_fusion × ffn_ratio
        dropout   : dropout probability
        has_carry : whether to accept a carry tensor from the previous level
    """

    def __init__(
        self,
        vggt_dim: int,
        jepa_dim: int,
        d_fusion: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        has_carry: bool = False,
    ):
        super().__init__()

        self.geo_proj  = nn.Linear(vggt_dim, d_fusion)
        self.sem_proj  = nn.Linear(jepa_dim, d_fusion)
        self.carry_proj = nn.Linear(d_fusion, d_fusion) if has_carry else None

        self.cross_attn = nn.MultiheadAttention(
            d_fusion, num_heads, dropout=dropout, batch_first=True
        )

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
        Q = self.geo_proj(vggt_feat)     # (B*S, 1369, D_f)
        K = self.sem_proj(jepa_feat)     # (B*S,  576, D_f)
        V = K

        if carry is not None and self.carry_proj is not None:
            Q = Q + self.carry_proj(carry)

        attn_out, _ = self.cross_attn(Q, K, V)   # (B*S, 1369, D_f)
        x     = self.norm1(Q + attn_out)          # residual + LN
        fused = x + self.ffn(self.norm2(x))       # FFN residual
        return fused


class MultiLevelFusion(nn.Module):
    """
    4-level cross-attention fusion with carry chain.

    carry chain:
        fused_0  ──carry──►  Q_1
        fused_1  ──carry──►  Q_2
        fused_2  ──carry──►  Q_3

    Args:
        num_levels : number of fusion levels (default 4)
        vggt_dim   : VGGT feature dim   (default 2048)
        jepa_dim   : JEPA feature dim   (default 1024)
        d_fusion   : output / attention dim (default 512)
        num_heads  : MHA heads (default 8)
        ffn_ratio  : FFN expansion ratio (default 4.0)
        dropout    : dropout probability (default 0.0)
    """

    def __init__(
        self,
        num_levels: int = 4,
        vggt_dim: int = 2048,
        jepa_dim: int = 1024,
        d_fusion: int = 512,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.levels = nn.ModuleList([
            CrossAttnLevel(
                vggt_dim=vggt_dim,
                jepa_dim=jepa_dim,
                d_fusion=d_fusion,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                has_carry=(i > 0),
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
            vg = vggt_feats[i].flatten(0, 1)    # (B*S, 1369, 2048)
            je = jepa_feats[i].flatten(0, 1)    # (B*S,  576, 1024)

            fused = level(vg, je, carry)         # (B*S, 1369, D_f)
            carry = fused

            fused_list.append(fused.view(B, S, fused.shape[-2], fused.shape[-1]))

        return fused_list   # list of 4 × (B, S, 1369, D_f)
