"""
Registry of pluggable "semantic" X-encoders (V-JEPA, I-JEPA, ... future ones).

Single source of truth for everything that differs between them: how to
build the unloaded backbone, which checkpoint key holds the frozen weights,
what image size/patch layout preprocessing must produce, and the resulting
token/dim shape. `encoders.FrozenSemanticEncoder` and `model.SingleEncoderXEncoder`
are both driven entirely by this table -- neither contains any
encoder-specific branching.

To add a new semantic encoder for an x_encoder_type ablation (config.py's
FusionConfig.x_encoder_type): add one EncoderSpec entry below. Nothing in
encoders.py, model.py, config.py, preprocess.py, or gvjepa_collate needs to
change -- they all resolve behavior by looking up SEMANTIC_ENCODERS[type].

Not covered here: FrozenVGGT (encoders.py) -- structurally different (VGGT's
own Aggregator, not a generic ViT backbone) and only ever used as the fixed
geometric stream in x_encoder_type="fusion_gv", not swappable via this
registry in the current architecture.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CKPT_DIR = _REPO_ROOT / "ckpts"


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    build_backbone: Callable[[], nn.Module]   # unloaded model, out_layers set by FrozenSemanticEncoder
    default_ckpt: str
    ckpt_state_key: str                       # e.g. "ema_encoder", "target_encoder"
    strip_prefixes: tuple[str, ...]           # stripped from state_dict keys, in order
    out_layers: tuple[int, ...]               # which backbone blocks to return features from
    img_size: int                             # preprocess target size (square)
    add_temporal_dim: bool                    # preprocess must add a T=1 dim (video-mode ViT) or not
    embed_dim: int                            # output channel dim
    num_patches: int                          # output spatial token count per frame


def _build_vjepa_backbone() -> nn.Module:
    from app.vjepa_2_1.models import vision_transformer as jepa_vit
    return jepa_vit.vit_large(
        patch_size=16,
        img_size=(384, 384),
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,    # T=1 -> image mode
        interpolate_rope=True,
        n_output_distillation=1,
    )


def _build_ijepa_backbone() -> nn.Module:
    from src.models.vision_transformer import vit_huge
    return vit_huge(patch_size=14, img_size=(224, 224), num_frames=1, tubelet_size=1)


SEMANTIC_ENCODERS: dict[str, EncoderSpec] = {
    "vjepa": EncoderSpec(
        name="vjepa",
        build_backbone=_build_vjepa_backbone,
        default_ckpt=str(_CKPT_DIR / "vjepa2_1_vitl_dist_vitG_384.pt"),
        ckpt_state_key="ema_encoder",
        strip_prefixes=("module.", "backbone."),
        out_layers=(5, 11, 17, 23),   # subset of hierarchical_layers for depth=24
        img_size=384,
        add_temporal_dim=True,
        embed_dim=1024,
        num_patches=576,   # 384/16 = 24 -> 24x24
    ),
    "ijepa": EncoderSpec(
        name="ijepa",
        build_backbone=_build_ijepa_backbone,
        default_ckpt=str(_CKPT_DIR / "IN1K-vit.h.14-300e.pth.tar"),
        ckpt_state_key="target_encoder",
        strip_prefixes=("module.",),
        out_layers=(7, 15, 23, 31),   # evenly spaced across depth=32, mirrors vjepa's relative spacing
        img_size=224,
        add_temporal_dim=False,
        embed_dim=1280,
        num_patches=256,   # 224/14 = 16 -> 16x16
    ),
}
