from dataclasses import dataclass, field
import os

_ROOT = os.path.dirname(os.path.dirname(__file__))


@dataclass
class FusionConfig:
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

    # ── Fusion module ──────────────────────────────────────────────────────────
    num_levels: int = 4
    d_fusion: int = 512
    num_heads: int = 8
    ffn_ratio: float = 4.0
    dropout: float = 0.0

    # ── Checkpoints ───────────────────────────────────────────────────────────
    vggt_ckpt: str = field(default_factory=lambda: os.path.join(_ROOT, "ckpts", "vggt.pt"))
    jepa_ckpt: str = field(
        default_factory=lambda: os.path.join(
            _ROOT, "ckpts", "vjepa2_1_vitl_dist_vitG_384.pt"
        )
    )
