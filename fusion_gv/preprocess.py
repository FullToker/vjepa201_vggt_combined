"""
Preprocessing for FusionGV.

Strategy: decode each image once to an intermediate tensor, then derive
both encoder inputs in-memory — avoids double disk I/O and double JPEG decode.

    raw image
        └── decode once  →  (3, H, W) float32 [0, 1]
                ├── _to_vggt(t)                        →  (3, 518, 518)  — normalisation done inside Aggregator
                ├── _to_semantic(t, 384, add_t_dim=True) →  (3, 1, 384, 384)  — ImageNet-normalised, T=1
                │     (need_jepa=True — the fixed V-JEPA shape fusion_gv's FusionGV always needs)
                └── _to_semantic(t, size, add_t_dim)    →  (3, [1,] size, size) — ImageNet-normalised
                      (semantic_img_size != None — generic, for whichever single-semantic-encoder
                      ablation is configured; caller supplies size/add_t_dim from
                      encoder_registry.SEMANTIC_ENCODERS[x_encoder_type], see gvjepa_collate)

This module never imports encoder_registry (keeps it dependency-light for
DataLoader workers) — callers resolve img_size/add_t_dim from the registry
themselves and pass them in.

Public API
----------
preprocess(images)  ->  (images_vggt, images_jepa)
    images_vggt  : (1, S, 3, 518, 518)   float32, [0, 1], or None
    images_jepa  : (S,  3, 1, 384, 384)  float32, ImageNet-normalised, or None
    images_semantic : (S, 3, [1,] size, size) float32, ImageNet-normalised, or None
                    (only when semantic_img_size is given; 3rd return value
                    otherwise omitted)
"""

from typing import List, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

_VGGT_SIZE = 518    # must be divisible by 14
_JEPA_SIZE = 384    # must be divisible by 16 -- fusion_gv's fixed V-JEPA shape

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── decode ─────────────────────────────────────────────────────────────────────

def _decode(src: Union[str, Image.Image]) -> torch.Tensor:
    """
    Load image, composite RGBA onto white if needed, convert to RGB.
    Resize so the short side is at least _VGGT_SIZE (largest required size).
    Returns (3, H, W) float32 in [0, 1].
    """
    img = Image.open(src) if isinstance(src, str) else src
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")

    w, h = img.size
    short = min(w, h)
    if short < _VGGT_SIZE:
        scale = _VGGT_SIZE / short
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)

    return TF.to_tensor(img)   # (3, H, W), float32 [0, 1]


# ── per-encoder crops (in-memory) ─────────────────────────────────────────────

def _to_vggt(t: torch.Tensor) -> torch.Tensor:
    """
    (3, H, W) [0,1]  →  (3, 518, 518) [0,1]
    Bilinear resize to short-side=518, then centre-crop.
    Normalisation is handled inside the VGGT Aggregator itself.
    """
    _, h, w = t.shape
    if min(h, w) != _VGGT_SIZE:
        scale = _VGGT_SIZE / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        t = F.interpolate(t.unsqueeze(0), size=(new_h, new_w),
                          mode="bilinear", align_corners=False).squeeze(0)
    return TF.center_crop(t, _VGGT_SIZE)   # (3, 518, 518)


def _to_semantic(t: torch.Tensor, img_size: int, add_t_dim: bool) -> torch.Tensor:
    """
    (3, H, W) [0,1]  →  (3, [1,] img_size, img_size) ImageNet-normalised
    Bilinear resize to short-side=img_size, centre-crop, ImageNet-normalise.
    add_t_dim=True appends a T=1 dim (video-mode ViTs, e.g. V-JEPA);
    add_t_dim=False leaves it as a plain image tensor (e.g. I-JEPA).
    Generic across every registry.SEMANTIC_ENCODERS entry -- one crop
    function for all of them, driven by the caller-supplied size/add_t_dim.
    """
    _, h, w = t.shape
    scale = img_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    t = F.interpolate(t.unsqueeze(0), size=(new_h, new_w),
                      mode="bilinear", align_corners=False).squeeze(0)
    t = TF.center_crop(t, img_size)        # (3, img_size, img_size)
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    return t.unsqueeze(1) if add_t_dim else t   # (3, 1, size, size) or (3, size, size)


# ── public API ─────────────────────────────────────────────────────────────────

def preprocess(
    images: List[Union[str, Image.Image]],
    *,
    need_vggt: bool = True,
    need_jepa: bool = True,
    semantic_img_size: int | None = None,
    semantic_add_t_dim: bool = False,
) -> tuple:
    """
    Decode each image once and derive requested encoder inputs in-memory.

    Args:
        images: list of S file paths or PIL Images
        need_vggt: build the VGGT input tensor
        need_jepa: build the fixed 384px V-JEPA input tensor (fusion_gv's
            FusionGV always needs exactly this shape for its semantic stream)
        semantic_img_size: build a 3rd generic tensor at this size for
            whichever single-semantic-encoder ablation is configured (see
            encoder_registry.SEMANTIC_ENCODERS[x_encoder_type].img_size) --
            None (default) omits the 3rd return value entirely
        semantic_add_t_dim: whether that 3rd tensor needs a T=1 dim (see
            encoder_registry.SEMANTIC_ENCODERS[x_encoder_type].add_temporal_dim)

    Returns:
        images_vggt     : (1, S, 3, 518, 518)  float32, [0, 1], or None
        images_jepa     : (S,  3, 1, 384, 384) float32, ImageNet-normalised, or None
        images_semantic : (S, 3, [1,] semantic_img_size, semantic_img_size)
                          float32, ImageNet-normalised
                          -- omitted from the return tuple unless semantic_img_size is set
    """
    need_semantic = semantic_img_size is not None
    if not need_vggt and not need_jepa and not need_semantic:
        raise ValueError("At least one of need_vggt, need_jepa, semantic_img_size must be set.")

    vggt_frames, jepa_frames, semantic_frames = [], [], []

    for src in images:
        raw = _decode(src)              # (3, H, W)  - one decode per image
        if need_vggt:
            vggt_frames.append(_to_vggt(raw))    # (3, 518, 518)
        if need_jepa:
            jepa_frames.append(_to_semantic(raw, _JEPA_SIZE, add_t_dim=True))   # (3, 1, 384, 384)
        if need_semantic:
            semantic_frames.append(_to_semantic(raw, semantic_img_size, semantic_add_t_dim))

    images_vggt = torch.stack(vggt_frames).unsqueeze(0) if need_vggt else None
    images_jepa = torch.stack(jepa_frames) if need_jepa else None

    if not need_semantic:
        return images_vggt, images_jepa

    images_semantic = torch.stack(semantic_frames)
    return images_vggt, images_jepa, images_semantic
