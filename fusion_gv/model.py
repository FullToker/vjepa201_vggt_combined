"""
FusionGV: semantic (V-JEPA 2.1) + geometric (VGGT) final-level feature fusion.

Both encoders are frozen. Only SingleLevelFusion parameters are trained.

Usage
-----
    from fusion_gv import FusionGV, FusionConfig, preprocess

    cfg   = FusionConfig()
    model = FusionGV(cfg).cuda()

    # preprocess accepts file paths or PIL Images
    images_vggt, images_jepa = preprocess(image_list)

    # move to device
    images_vggt = images_vggt.cuda()
    images_jepa = images_jepa.cuda()

    fused = model(images_vggt, images_jepa)
    # fused: (1, S, 1369, 2048)   [final-level LN+MLP projector + spatial align + concat]
"""

import torch
import torch.nn as nn

from fusion_gv.config import FusionConfig
from fusion_gv.encoders import FrozenVGGT, FrozenJEPA
from fusion_gv.fusion_aligned import SingleLevelFusion

_X_ENCODER_TYPES = ("fusion_gv", "vjepa")


def _build_fusion(config: FusionConfig) -> nn.Module:
    return SingleLevelFusion(
        vggt_dim=config.vggt_out_dim,
        jepa_dim=config.jepa_embed_dim,
        proj_dim=config.proj_dim,
        src_grid=int(config.jepa_num_patches ** 0.5),   # 24
        tgt_grid=int(config.vggt_num_patches ** 0.5),   # 37
    )


class VJEPAOnlyXEncoder(nn.Module):
    """V-JEPA-only X-encoder with the same output contract as FusionGV."""

    def __init__(self, config: FusionConfig | None = None):
        super().__init__()
        if config is None:
            config = FusionConfig(x_encoder_type="vjepa")
        self.config = config
        self.jepa_encoder = FrozenJEPA(config.jepa_ckpt)

    def forward(
        self,
        images_vggt: torch.Tensor | None,
        images_jepa: torch.Tensor,
    ) -> torch.Tensor:
        """Return final-level V-JEPA features: (B, S, 576, 1024)."""
        if images_vggt is not None:
            B, S = images_vggt.shape[:2]
        else:
            B = 1
            S = images_jepa.shape[0]
        return self.jepa_encoder(images_jepa, B, S)[-1]

    def trainable_parameters(self):
        return iter(())


def build_x_encoder(config: FusionConfig | None = None) -> nn.Module:
    """Build the configured visual X-encoder."""
    if config is None:
        config = FusionConfig()
    if config.x_encoder_type == "fusion_gv":
        return FusionGV(config)
    if config.x_encoder_type == "vjepa":
        return VJEPAOnlyXEncoder(config)
    raise ValueError(
        f"Unknown x_encoder_type '{config.x_encoder_type}'. "
        f"Choose from {_X_ENCODER_TYPES}."
    )


class FusionGV(nn.Module):
    """
    Top-level model.

    Frozen encoders (only the final level of each is used):
        FrozenVGGT  — geometric features, final level (B, S, 1369, 2048)
        FrozenJEPA  — semantic features,  final level (B, S,  576, 1024)

    Fusion module: SingleLevelFusion (per-stream LN+MLP projector,
    bilinear spatial align, channel concat). Output dim = 2 * config.proj_dim
    (default 2048).

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

        self.fusion = _build_fusion(config)

    def forward(
        self,
        images_vggt: torch.Tensor,
        images_jepa: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images_vggt : (B, S, 3, 518, 518)  float32 [0, 1]
            images_jepa : (B*S, 3, 1, 384, 384) float32, ImageNet-normalised

        Returns:
            (B, S, 1369, D_fused)   D_fused = 2 * config.proj_dim (default 2048)
        """
        B, S = images_vggt.shape[:2]

        vggt_feats = self.vggt_encoder(images_vggt)          # 4 × (B, S, 1369, 2048)
        jepa_feats = self.jepa_encoder(images_jepa, B, S)    # 4 × (B, S,  576, 1024)

        return self.fusion(vggt_feats[-1], jepa_feats[-1])    # (B, S, 1369, D_f), final level only

    def trainable_parameters(self):
        """Returns trainable parameters (fusion module only; encoders are frozen)."""
        return self.fusion.parameters()
