"""
Preprocessing for FusionGV.

Strategy: decode each image once to an intermediate tensor, then derive
both encoder inputs in-memory — avoids double disk I/O and double JPEG decode.

    raw image
        └── decode once  →  (3, H, W) float32 [0, 1]
                ├── _to_vggt(t)   →  (3, 518, 518)     — normalisation done inside Aggregator
                ├── _to_jepa(t)   →  (3, 1, 384, 384)  — ImageNet-normalised, T=1
                └── _to_ijepa(t)  →  (3, 224, 224)     — ImageNet-normalised, no T dim
                                      (need_ijepa=True, off by default — only
                                      needed for x_encoder_type == "ijepa")

Public API
----------
preprocess(images)  ->  (images_vggt, images_jepa)
    images_vggt  : (1, S, 3, 518, 518)   float32, [0, 1], or None
    images_jepa  : (S,  3, 1, 384, 384)  float32, ImageNet-normalised, or None
    images_ijepa : (S,  3, 224, 224)     float32, ImageNet-normalised, or None
                    (only when need_ijepa=True; 3rd return value otherwise omitted)
"""

from typing import List, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

_VGGT_SIZE = 518    # must be divisible by 14
_JEPA_SIZE = 384    # must be divisible by 16
_IJEPA_SIZE = 224   # must be divisible by 14 (I-JEPA ViT-H/14)

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


def _to_jepa(t: torch.Tensor) -> torch.Tensor:
    """
    (3, H, W) [0,1]  →  (3, 1, 384, 384) ImageNet-normalised
    Bilinear resize to short-side=384, centre-crop, add T=1 dim.
    """
    _, h, w = t.shape
    scale = _JEPA_SIZE / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    t = F.interpolate(t.unsqueeze(0), size=(new_h, new_w),
                      mode="bilinear", align_corners=False).squeeze(0)
    t = TF.center_crop(t, _JEPA_SIZE)      # (3, 384, 384)
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    return t.unsqueeze(1)                  # (3, 1, 384, 384)  ← T=1


def _to_ijepa(t: torch.Tensor) -> torch.Tensor:
    """
    (3, H, W) [0,1]  →  (3, 224, 224) ImageNet-normalised
    Bilinear resize to short-side=224, centre-crop. No T dim -- I-JEPA's
    VisionTransformer takes plain (B, 3, H, W) image input.
    """
    _, h, w = t.shape
    scale = _IJEPA_SIZE / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    t = F.interpolate(t.unsqueeze(0), size=(new_h, new_w),
                      mode="bilinear", align_corners=False).squeeze(0)
    t = TF.center_crop(t, _IJEPA_SIZE)     # (3, 224, 224)
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD


# ── public API ─────────────────────────────────────────────────────────────────

def preprocess(
    images: List[Union[str, Image.Image]],
    *,
    need_vggt: bool = True,
    need_jepa: bool = True,
    need_ijepa: bool = False,
) -> tuple:
    """
    Decode each image once and derive requested encoder inputs in-memory.

    Args:
        images: list of S file paths or PIL Images
        need_vggt: build the VGGT input tensor
        need_jepa: build the V-JEPA input tensor
        need_ijepa: build the I-JEPA input tensor (x_encoder_type == "ijepa"
            ablation only -- off by default, adds a 3rd return value when set)

    Returns:
        images_vggt  : (1, S, 3, 518, 518)  float32, [0, 1], or None
        images_jepa  : (S,  3, 1, 384, 384) float32, ImageNet-normalised, or None
        images_ijepa : (S,  3, 224, 224)    float32, ImageNet-normalised, or None
                        -- omitted from the return tuple unless need_ijepa=True
    """
    if not need_vggt and not need_jepa and not need_ijepa:
        raise ValueError("At least one of need_vggt, need_jepa, need_ijepa must be true.")

    vggt_frames, jepa_frames, ijepa_frames = [], [], []

    for src in images:
        raw = _decode(src)              # (3, H, W)  - one decode per image
        if need_vggt:
            vggt_frames.append(_to_vggt(raw))    # (3, 518, 518)
        if need_jepa:
            jepa_frames.append(_to_jepa(raw))    # (3, 1, 384, 384)
        if need_ijepa:
            ijepa_frames.append(_to_ijepa(raw))  # (3, 224, 224)

    images_vggt = torch.stack(vggt_frames).unsqueeze(0) if need_vggt else None
    images_jepa = torch.stack(jepa_frames) if need_jepa else None

    if not need_ijepa:
        return images_vggt, images_jepa

    images_ijepa = torch.stack(ijepa_frames)
    return images_vggt, images_jepa, images_ijepa
