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

import numpy as np
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
    assert FusionConfig(x_encoder_type="ijepa").visual_dim == 1280
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


# ── Phase 3b: frame-block shuffle (random tensors, no weights) ────────────────
print("\n=== Phase 3b: shuffle_frame_blocks (random tensors, no weights) ===")

from fusion_gv.gvjepa import shuffle_frame_blocks

def _test_shuffle_frame_blocks():
    Bs, Sf, K, D = 6, 8, 2, 5
    x = torch.arange(Bs * Sf * K * D, dtype=torch.float32).reshape(Bs, Sf * K, D)   # all values unique
    assert shuffle_frame_blocks(x, Sf, 0.0) is x, "prob=0 must be a no-op"
    y = shuffle_frame_blocks(x, Sf, 1.0)
    assert y.shape == x.shape
    xb, yb = x.reshape(Bs, Sf, K, D), y.reshape(Bs, Sf, K, D)
    for b in range(Bs):
        # every output block is one whole input block (K tokens intact, in order), each used exactly once
        src = [(xb[b] == yb[b, i]).all(dim=(1, 2)).nonzero().item() for i in range(Sf)]
        assert sorted(src) == list(range(Sf)), src
    assert not torch.equal(y, x), "prob=1 left every sample unchanged"
    big = torch.arange(200 * Sf * K * D, dtype=torch.float32).reshape(200, Sf * K, D)
    changed = (shuffle_frame_blocks(big, Sf, 0.5) != big).flatten(1).any(dim=1).sum().item()
    assert 40 < changed < 160, f"prob=0.5 gate off: {changed}/200 samples changed"
    return changed

r3b = check("shuffle_frame_blocks: identity at 0, whole-block permutation at 1, per-sample gate", _test_shuffle_frame_blocks)
if r3b is not None:
    print(f"         prob=0.5 changed {r3b}/200 samples")


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


# ── Phase 5: single-semantic-encoder X-encoder forward (registry-driven) ─────
# Loops over every encoder_registry.SEMANTIC_ENCODERS entry -- adding a new
# spec there gets it covered here automatically, no new test block needed.
print("\n=== Phase 5: single-semantic-encoder X-encoder forward (registry-driven) ===")

from fusion_gv.encoder_registry import SEMANTIC_ENCODERS
from fusion_gv.model import SingleEncoderXEncoder
from fusion_gv.gvjepa import FusionGVJEPA, GVJEPAConfig

for enc_name, spec in SEMANTIC_ENCODERS.items():
    print(f"--- {enc_name} ---")
    if not os.path.exists(spec.default_ckpt):
        print(f"  [{SKIP}] checkpoint not found: {spec.default_ckpt}")
        continue

    def _test_single_encoder_xencoder(enc_name=enc_name, spec=spec):
        cfg = FusionConfig(x_encoder_type=enc_name)
        model = SingleEncoderXEncoder(cfg).eval()
        imgs = [Image.new("RGB", (640, 480)) for _ in range(S)]
        vggt_t, _, sem_t = preprocess(
            imgs, semantic_img_size=spec.img_size, semantic_add_t_dim=spec.add_temporal_dim
        )
        with torch.no_grad():
            out = model(vggt_t, sem_t)
        assert out.shape == (1, S, spec.num_patches, spec.embed_dim), out.shape
        return out.shape

    def _test_gvjepa_with_single_encoder(enc_name=enc_name, spec=spec):
        fusion_cfg = FusionConfig(x_encoder_type=enc_name)
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
        vggt_t, _, sem_t = preprocess(
            imgs, semantic_img_size=spec.img_size, semantic_add_t_dim=spec.add_temporal_dim
        )
        with torch.no_grad():
            out = model(vggt_t, sem_t, queries=["describe scene"], targets=["a scene"])
        assert out["pred"].shape == (1, 64), out["pred"].shape
        assert out["target"].shape == (1, 64), out["target"].shape
        return out["pred"].shape

    def _test_frame_shuffle_model(enc_name=enc_name, spec=spec):
        model_cfg = GVJEPAConfig(
            fusion=FusionConfig(x_encoder_type=enc_name),
            predictor_hidden_size=128,
            predictor_layers=1,
            predictor_heads=4,
            shared_embed_dim=64,
            query_model_name="toy",
            y_encoder_name="toy",
        )
        model = FusionGVJEPA(model_cfg).eval()
        # distinct frames: identical images would make any frame permutation a trivial no-op
        rng = np.random.default_rng(0)
        imgs = [Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)) for _ in range(S)]
        vggt_t, _, sem_t = preprocess(
            imgs, semantic_img_size=spec.img_size, semantic_add_t_dim=spec.add_temporal_dim
        )

        def pred():
            with torch.no_grad():
                return model(vggt_t, sem_t, queries=["describe scene"], targets=["a scene"])["pred"]

        base = pred()
        # eval mode reads frame_shuffle_eval_prob; frame_shuffle_prob (train) must not leak in
        model.config.frame_shuffle_prob = 1.0
        assert torch.allclose(pred(), base, atol=1e-4), "train-mode knob affected eval"
        model.config.frame_shuffle_prob = 0.0
        model.config.frame_shuffle_eval_prob = 1.0
        assert not torch.allclose(pred(), base, atol=1e-4), "eval shuffle before frame_embed changed nothing"
        # train mode reads frame_shuffle_prob
        model.config.frame_shuffle_eval_prob = 0.0
        model.config.frame_shuffle_prob = 1.0
        model.train()
        assert not torch.allclose(pred(), base, atol=1e-4), "train shuffle before frame_embed changed nothing"
        model.eval()
        # frame_embed is the only frame-order channel: zero it and shuffling must be a no-op
        model.config.frame_shuffle_prob = 0.0
        with torch.no_grad():
            model.frame_embed.weight.zero_()
        zeroed_base = pred()
        model.config.frame_shuffle_eval_prob = 1.0
        assert torch.allclose(pred(), zeroed_base, atol=1e-4), "predictor is order-sensitive beyond frame_embed"
        return True

    r_enc = check(
        f"SingleEncoderXEncoder('{enc_name}') forward → (B,S,{spec.num_patches},{spec.embed_dim})",
        _test_single_encoder_xencoder,
    )
    check(f"frame shuffle on x_encoder_type='{enc_name}': knobs + frame_embed is the only order channel",
          _test_frame_shuffle_model)
    r_gvjepa = check(f"FusionGVJEPA x_encoder_type='{enc_name}' forward", _test_gvjepa_with_single_encoder)
    if r_enc:
        print(f"         output shape     : {r_enc}")
    if r_gvjepa:
        print(f"         pred embedding   : {r_gvjepa}")


# ── Phase 6: VGGT-only X-encoder forward (no semantic stream at all) ─────────
print("\n=== Phase 6: VGGT-only X-encoder forward (requires VGGT ckpt) ===")

if not os.path.exists(vggt_ckpt):
    print(f"  [{SKIP}] checkpoint not found: ckpts/vggt.pt")
    print(f"           run: python download_ckpts.py")
else:
    from fusion_gv.model import VGGTOnlyXEncoder

    def _test_vggt_only_xencoder():
        cfg = FusionConfig(x_encoder_type="vggt")
        model = VGGTOnlyXEncoder(cfg).eval()
        imgs = [Image.new("RGB", (640, 480)) for _ in range(S)]
        vggt_t, _ = preprocess(imgs, need_jepa=False)
        with torch.no_grad():
            out = model(vggt_t)
        assert out.shape == (1, S, 1369, 2048), out.shape
        return out.shape

    def _test_gvjepa_with_vggt_only():
        fusion_cfg = FusionConfig(x_encoder_type="vggt")
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
        vggt_t, _ = preprocess(imgs, need_jepa=False)
        with torch.no_grad():
            # images_jepa=None -- no semantic stream, VGGTOnlyXEncoder ignores it
            out = model(vggt_t, None, queries=["describe scene"], targets=["a scene"])
        assert out["pred"].shape == (1, 64), out["pred"].shape
        assert out["target"].shape == (1, 64), out["target"].shape
        return out["pred"].shape

    r9 = check("VGGTOnlyXEncoder forward → (B,S,1369,2048)", _test_vggt_only_xencoder)
    r10 = check("FusionGVJEPA x_encoder_type='vggt' forward (images_jepa=None)", _test_gvjepa_with_vggt_only)
    if r9:
        print(f"         output shape     : {r9}")
    if r10:
        print(f"         pred embedding   : {r10}")


print("\n=== done ===\n")
