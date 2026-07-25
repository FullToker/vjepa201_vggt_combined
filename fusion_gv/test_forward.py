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

import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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

def _test_encoder_dim_config():
    FusionConfig = __import__("fusion_gv.config", fromlist=["FusionConfig"]).FusionConfig
    assert FusionConfig(x_encoder_type="fusion_gv").visual_dim == 2048
    assert FusionConfig(x_encoder_type="vjepa").visual_dim == 1024
    assert FusionConfig(x_encoder_type="vjepa", x_encoder_output_dim=768).visual_dim == 768

check("FusionConfig x_encoder_type / x_encoder_output_dim", _test_encoder_dim_config)
check("import SingleLevelFusion",
      lambda: __import__("fusion_gv.fusion_aligned", fromlist=["SingleLevelFusion"]))
check("import preprocess",
      lambda: __import__("fusion_gv.preprocess", fromlist=["preprocess"]))
check("import build_x_encoder",
      lambda: __import__("fusion_gv.model", fromlist=["build_x_encoder"]))


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

from fusion_gv.fusion_aligned import SingleLevelFusion

B, S = 1, 8
D_fused = 2048   # 2 * proj_dim (default proj_dim=1024)

def _make_feats():
    vggt = torch.randn(B, S, 1369, 2048)   # final level only
    jepa = torch.randn(B, S,  576, 1024)   # final level only
    return vggt, jepa

def _test_single_level_fusion():
    m = SingleLevelFusion()
    vggt, jepa = _make_feats()
    out = m(vggt, jepa)
    assert out.shape == (B, S, 1369, D_fused), f"{out.shape}"
    return out.shape

r1 = check("SingleLevelFusion  (LN+MLP+align+concat)  → (B,S,1369,2048)", _test_single_level_fusion)
if r1:
    print(f"         output shape : {r1}")


# ── Phase 4: full FusionGV (requires ckpts/) ──────────────────────────────────
print("\n=== Phase 4: full FusionGV forward (requires ckpts/) ===")

from fusion_gv.config import FusionConfig
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

    def _test_full_fusion():
        cfg = FusionConfig()
        model = FusionGV(cfg).eval()
        imgs = [Image.new("RGB", (640, 480)) for _ in range(S)]
        vggt_t, jepa_t = preprocess(imgs)
        with torch.no_grad():
            out = model(vggt_t, jepa_t)
        assert out.shape == (1, S, 1369, D_fused), out.shape
        return out.shape

    r3 = check("FusionGV end-to-end", _test_full_fusion)
    if r3:
        print(f"         output shape : {r3}")


# ── Phase 5: V-JEPA-only X-encoder forward ───────────────────────────────────
print("\n=== Phase 5: V-JEPA-only X-encoder forward (requires V-JEPA ckpt) ===")

if not os.path.exists(jepa_ckpt):
    print(f"  [{SKIP}] checkpoint not found: ckpts/vjepa2_1_vitl_dist_vitG_384.pt")
    print(f"           run: python download_ckpts.py")
else:
    from fusion_gv.model import VJEPAOnlyXEncoder
    from fusion_gv.gvjepa import FusionGVJEPA, GVJEPAConfig

    def _test_vjepa_only_xencoder():
        cfg = FusionConfig(x_encoder_type="vjepa")
        model = VJEPAOnlyXEncoder(cfg).eval()
        imgs = [Image.new("RGB", (640, 480)) for _ in range(S)]
        vggt_t, jepa_t = preprocess(imgs)
        with torch.no_grad():
            out = model(vggt_t, jepa_t)
        assert out.shape == (1, S, 576, 1024), out.shape
        return out.shape

    def _test_gvjepa_with_vjepa_xencoder():
        fusion_cfg = FusionConfig(x_encoder_type="vjepa")
        model_cfg = GVJEPAConfig(
            fusion=fusion_cfg,
            predictor_hidden_size=128,
            predictor_layers=1,
            predictor_heads=4,
            shared_embed_dim=64,
            query_model_name="toy",
            y_encoder_name="toy",
        )
        model = FusionGVJEPA(model_cfg).eval()
        imgs = [Image.new("RGB", (640, 480)) for _ in range(S)]
        vggt_t, jepa_t = preprocess(imgs)
        with torch.no_grad():
            out = model(vggt_t, jepa_t, queries=["describe scene"], targets=["a scene"])
        assert out["pred"].shape == (1, 64), out["pred"].shape
        assert out["target"].shape == (1, 64), out["target"].shape
        return out["pred"].shape

    r5 = check("VJEPAOnlyXEncoder forward → (B,S,576,1024)", _test_vjepa_only_xencoder)
    r6 = check("FusionGVJEPA x_encoder_type='vjepa' forward", _test_gvjepa_with_vjepa_xencoder)
    if r5:
        print(f"         output shape     : {r5}")
    if r6:
        print(f"         pred embedding   : {r6}")


print("\n=== done ===\n")
