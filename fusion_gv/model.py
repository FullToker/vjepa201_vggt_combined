"""
FusionGV: semantic (V-JEPA 2.1) + geometric (VGGT) multi-level feature fusion.

Both encoders are frozen. Only MultiLevelFusion parameters are trained.

Usage
-----
    from fusion_gv import FusionGV, FusionConfig, preprocess

    cfg   = FusionConfig(d_fusion=512)
    model = FusionGV(cfg).cuda()

    # preprocess accepts file paths or PIL Images
    images_vggt, images_jepa = preprocess(image_list)

    # move to device
    images_vggt = images_vggt.cuda()
    images_jepa = images_jepa.cuda()

    fused = model(images_vggt, images_jepa)
    # fused: list of 4 × (1, S, 1369, 512)
"""

import torch
import torch.nn as nn

from fusion_gv.config import FusionConfig
from fusion_gv.encoders import FrozenVGGT, FrozenJEPA
from fusion_gv.fusion import MultiLevelFusion


class FusionGV(nn.Module):
    """
    Top-level model.

    Frozen encoders:
        FrozenVGGT  — geometric features, 4 levels × (B, S, 1369, 2048)
        FrozenJEPA  — semantic features,  4 levels × (B, S,  576, 1024)

    Trainable module:
        MultiLevelFusion — cross-attention fusion, 4 levels × (B, S, 1369, D_f)

    Args:
        config : FusionConfig instance (uses defaults if None)
    """

    def __init__(self, config: FusionConfig | None = None):
        super().__init__()

        if config is None:
            config = FusionConfig()
        self.config = config

        self.vggt_encoder = FrozenVGGT(config.vggt_ckpt)
        self.jepa_encoder = FrozenJEPA(config.jepa_ckpt)

        self.fusion = MultiLevelFusion(
            num_levels=config.num_levels,
            vggt_dim=config.vggt_out_dim,
            jepa_dim=config.jepa_embed_dim,
            d_fusion=config.d_fusion,
            num_heads=config.num_heads,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
        )

    def forward(
        self,
        images_vggt: torch.Tensor,
        images_jepa: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Args:
            images_vggt : (B, S, 3, 518, 518)  float32 [0, 1]
            images_jepa : (B*S, 3, 1, 384, 384) float32, ImageNet-normalised

        Returns:
            list of 4 × (B, S, 1369, D_f)
            index 0 = early features, index 3 = final features
        """
        B, S = images_vggt.shape[:2]

        vggt_feats = self.vggt_encoder(images_vggt)          # 4 × (B, S, 1369, 2048)
        jepa_feats = self.jepa_encoder(images_jepa, B, S)    # 4 × (B, S,  576, 1024)

        return self.fusion(vggt_feats, jepa_feats)            # 4 × (B, S, 1369, D_f)

    def trainable_parameters(self):
        """Returns only the fusion module parameters (encoders are frozen)."""
        return self.fusion.parameters()
