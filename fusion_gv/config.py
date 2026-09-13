from dataclasses import dataclass, field
import os
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(__file__))


@dataclass
class FusionConfig:
    # X-encoder mode used by the training stack.
    # "fusion_gv" keeps the current VGGT + V-JEPA concat encoder.
    # "vjepa" uses V-JEPA alone as the X-encoder.
    # "ijepa" uses I-JEPA (image-pretrained, no video) alone as the X-encoder
    #   -- ablation counterpart to "vjepa", isolating video pretraining.
    x_encoder_type: str = "fusion_gv"
    x_encoder_output_dim: Optional[int] = None

    # ── VGGT (geometric encoder) ───────────────────────────────────────────────
    vggt_img_size: int = 518
    vggt_patch_size: int = 14               # 518 / 14 = 37
    vggt_embed_dim: int = 1024
    vggt_out_dim: int = 2048                # frame_inter ∥ global_inter → 2 × embed_dim
    vggt_num_patches: int = 1369            # 37 × 37
    vggt_patch_start_idx: int = 5           # 1 camera + 4 register tokens
    vggt_cached_rounds: tuple = (4, 11, 17, 23)

    # ── V-JEPA 2.1 ViT-L (semantic encoder) ───────────────────────────────────
    jepa_img_size: int = 384
    jepa_patch_size: int = 16               # 384 / 16 = 24
    jepa_embed_dim: int = 1024
    jepa_num_patches: int = 576             # 24 × 24
    jepa_out_layers: tuple = (5, 11, 17, 23)   # must be subset of hierarchical_layers

    # ── I-JEPA ViT-H/14 (image-pretrained ablation for jepa_encoder) ──────────
    ijepa_img_size: int = 224
    ijepa_patch_size: int = 14              # 224 / 14 = 16
    ijepa_embed_dim: int = 1280
    ijepa_num_patches: int = 256            # 16 × 16
    ijepa_out_layers: tuple = (7, 15, 23, 31)

    # ── Fusion module ──────────────────────────────────────────────────────────
    # SingleLevelFusion: per-stream LayerNorm + 2-layer MLP(GELU) projector,
    # bilinear spatial align (JEPA 24×24 → VGGT 37×37), channel concat.
    # Uses only the final level of each encoder. Output dim = 2 * proj_dim.
    proj_dim: int = 1024

    @property
    def fused_dim(self) -> int:
        """Output channel dim of the fusion module."""
        if self.x_encoder_output_dim is not None:
            return self.x_encoder_output_dim
        return self.proj_dim * 2   # 2048 by default

    @property
    def visual_dim(self) -> int:
        """Output channel dim produced by the configured X-encoder."""
        if self.x_encoder_output_dim is not None:
            return self.x_encoder_output_dim
        if self.x_encoder_type == "fusion_gv":
            return self.proj_dim * 2
        if self.x_encoder_type == "vjepa":
            return self.jepa_embed_dim
        if self.x_encoder_type == "ijepa":
            return self.ijepa_embed_dim
        raise ValueError(
            f"Unknown x_encoder_type '{self.x_encoder_type}'. "
            "Choose 'fusion_gv', 'vjepa', or 'ijepa'."
        )

    @property
    def visual_num_patches(self) -> int:
        """Spatial token count produced by the configured X-encoder."""
        if self.x_encoder_type == "fusion_gv":
            return self.vggt_num_patches
        if self.x_encoder_type == "vjepa":
            return self.jepa_num_patches
        if self.x_encoder_type == "ijepa":
            return self.ijepa_num_patches
        raise ValueError(
            f"Unknown x_encoder_type '{self.x_encoder_type}'. "
            "Choose 'fusion_gv', 'vjepa', or 'ijepa'."
        )

    # ── Checkpoints ───────────────────────────────────────────────────────────
    vggt_ckpt: str = field(default_factory=lambda: os.path.join(_ROOT, "ckpts", "vggt.pt"))
    jepa_ckpt: str = field(
        default_factory=lambda: os.path.join(
            _ROOT, "ckpts", "vjepa2_1_vitl_dist_vitG_384.pt"
        )
    )
    ijepa_ckpt: str = field(
        default_factory=lambda: os.path.join(_ROOT, "ckpts", "IN1K-vit.h.14-300e.pth.tar")
    )
