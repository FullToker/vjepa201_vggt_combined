"""
Forward pass validation for fusion_gv.

Runs three phases — each phase is independent and can pass/fail on its own.

Phase 1 : imports + config
Phase 2 : preprocess  (synthetic PIL image, no weights needed)
Phase 3 : fusion module shapes  (random tensors, no weights needed)
Phase 4 : full FusionGV forward  (requires ckpts/, skipped if missing)

Run from project root:
    python fusion_gv/test_forward.py
"""

import sys
import traceback

import torch
from PIL import Image

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


def check(name: str, fn):
    try:
        result = fn()
        print(f"  [{PASS}] {name}")
        return result
    except Exception as e:
        print(f"  [{FAIL}] {name}")
        traceback.print_exc()
        return None


# ── Phase 1: imports ───────────────────────────────────────────────────────────
print("\n=== Phase 1: imports + config ===")

cfg = check("import FusionConfig", lambda: (
    __import__("fusion_gv.config", fromlist=["FusionConfig"]).FusionConfig()
))
check("import MultiLevelFusion",
      lambda: __import__("fusion_gv.fusion", fromlist=["MultiLevelFusion"]))
check("import AlignedMultiLevelFusion",
      lambda: __import__("fusion_gv.fusion_aligned", fromlist=["AlignedMultiLevelFusion"]))
check("import preprocess",
      lambda: __import__("fusion_gv.preprocess", fromlist=["preprocess"]))


# ── Phase 2: preprocess ────────────────────────────────────────────────────────
print("\n=== Phase 2: preprocess (synthetic image, no weights) ===")

from fusion_gv.preprocess import preprocess

def _test_preprocess():
    imgs = [Image.new("RGB", (640, 480), color=(128, 64, 32)) for _ in range(3)]
    vggt, jepa = preprocess(imgs)
    assert vggt.shape == (1, 3, 3, 518, 518), f"vggt shape {vggt.shape}"
    assert jepa.shape == (3, 3, 1, 384, 384), f"jepa shape {jepa.shape}"
    assert vggt.min() >= 0.0 and vggt.max() <= 1.0, "vggt not in [0,1]"
    return vggt.shape, jepa.shape

result = check("preprocess 3 images → correct shapes", _test_preprocess)
if result:
    print(f"         images_vggt : {result[0]}")
    print(f"         images_jepa : {result[1]}")


# ── Phase 3: fusion module shapes ─────────────────────────────────────────────
print("\n=== Phase 3: fusion module shapes (random tensors, no weights) ===")

from fusion_gv.fusion import MultiLevelFusion
from fusion_gv.fusion_aligned import AlignedMultiLevelFusion

B, S, D_f = 1, 8, 512

def _make_feats():
    vggt = [torch.randn(B, S, 1369, 2048) for _ in range(4)]
    jepa = [torch.randn(B, S,  576, 1024) for _ in range(4)]
    return vggt, jepa

def _test_aligned():
    m = AlignedMultiLevelFusion(d_fusion=D_f)
    vggt, jepa = _make_feats()
    out = m(vggt, jepa)
    assert len(out) == 4
    for i, t in enumerate(out):
        assert t.shape == (B, S, 1369, D_f), f"level {i}: {t.shape}"
    return [t.shape for t in out]

def _test_cross_attn():
    m = MultiLevelFusion(d_fusion=D_f, num_heads=8)
    vggt, jepa = _make_feats()
    out = m(vggt, jepa)
    assert len(out) == 4
    for i, t in enumerate(out):
        assert t.shape == (B, S, 1369, D_f), f"level {i}: {t.shape}"
    return [t.shape for t in out]

r1 = check("AlignedMultiLevelFusion  (add)        4 levels → (B,S,1369,512)", _test_aligned)
r2 = check("MultiLevelFusion         (cross_attn) 4 levels → (B,S,1369,512)", _test_cross_attn)
if r1:
    print(f"         output per level : {r1[0]}")


# ── Phase 4: full FusionGV (requires ckpts/) ──────────────────────────────────
print("\n=== Phase 4: full FusionGV forward (requires ckpts/) ===")

import os
from fusion_gv.config import FusionConfig

ROOT = os.path.dirname(os.path.dirname(__file__))
vggt_ckpt = os.path.join(ROOT, "ckpts", "vggt.pt")
jepa_ckpt = os.path.join(ROOT, "ckpts", "vjepa2_1_vitl_dist_vitG_384.pt")

if not os.path.exists(vggt_ckpt) or not os.path.exists(jepa_ckpt):
    missing = []
    if not os.path.exists(vggt_ckpt): missing.append("ckpts/vggt.pt")
    if not os.path.exists(jepa_ckpt): missing.append("ckpts/vjepa2_1_vitl_dist_vitG_384.pt")
    print(f"  [{SKIP}] checkpoints not found: {', '.join(missing)}")
    print(f"           run: python download_ckpts.py")
else:
    from fusion_gv.model import FusionGV

    def _test_full_add():
        cfg = FusionConfig(d_fusion=D_f, fusion_type="add")
        model = FusionGV(cfg).eval()
        imgs = [Image.new("RGB", (640, 480)) for _ in range(S)]
        vggt_t, jepa_t = preprocess(imgs)
        with torch.no_grad():
            out = model(vggt_t, jepa_t)
        assert len(out) == 4
        for t in out:
            assert t.shape == (1, S, 1369, D_f)
        return out[0].shape

    def _test_full_cross():
        cfg = FusionConfig(d_fusion=D_f, fusion_type="cross_attn")
        model = FusionGV(cfg).eval()
        imgs = [Image.new("RGB", (640, 480)) for _ in range(S)]
        vggt_t, jepa_t = preprocess(imgs)
        with torch.no_grad():
            out = model(vggt_t, jepa_t)
        assert len(out) == 4
        for t in out:
            assert t.shape == (1, S, 1369, D_f)
        return out[0].shape

    r3 = check("FusionGV fusion_type='add'        end-to-end", _test_full_add)
    r4 = check("FusionGV fusion_type='cross_attn' end-to-end", _test_full_cross)
    if r3:
        print(f"         output per level : {r3}")


print("\n=== done ===\n")
