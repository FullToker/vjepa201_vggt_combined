"""
One-off cross-repo weight import: warm-start a fresh FusionGVJEPA's predictor
and y_encoder from an open-vljepa (../open-vljepa) trained checkpoint, before
running fusion_gv's own SFT.

Not part of fusion_gv's own save/resume lifecycle (see gvjepa_trainer.py
_save_ckpt / infer_gvjepa.py _load_checkpoint for that — same architecture,
same key names, strict=True). This is the opposite case: a *different*
architecture's checkpoint, only partially compatible, used once to seed a
brand-new model before training starts.

Why this transfers at all
--------------------------
Both repos build the predictor as the last N decoder layers of the same
pretrained meta-llama/Llama-3.2-1B checkpoint, and both use
google/embeddinggemma-300m as y_encoder. Those weights are shape-identical
and transfer directly. What does NOT transfer:
  - modality_embed / frame_embed / grounding_head: fusion_gv-only modules,
    open-vljepa has no equivalent — left at random init.
  - vis_proj: only shape-compatible when fusion.x_encoder_type == "vjepa"
    (visual_dim=1024, matching open-vljepa's single V-JEPA2 encoder). Under
    x_encoder_type == "fusion_gv" (2048-d VGGT+JEPA fusion) it's skipped.
  - predictor.embed_tokens: frozen on both sides, already loaded from the
    same HF checkpoint at construction time — intentionally not copied.

Key mapping (open-vljepa -> fusion_gv)
---------------------------------------
    predictor.layers.*     -> predictor.layers.*
    predictor.norm.*       -> predictor.norm.*
    predictor.vis_proj.*   -> vis_proj.*        (shape-gated, see above)
    predictor.out_proj.*   -> pred_proj.*
    y_encoder.model.*      -> y_encoder.*
    y_encoder.projector.*  -> y_proj.*

Usage
-----
    from fusion_gv.load_vljepa_init import load_predictor_and_y_encoder_from_vljepa

    model = build_model_from_config(cfg)
    load_predictor_and_y_encoder_from_vljepa(model, "checkpoints_msrvtt/best.pt")
    # ... continue into fusion_gv's normal SFT loop
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _load_matching(dst: nn.Module, src_sd: dict, tag: str) -> None:
    """Copy tensors from src_sd into dst's state_dict, name+shape must match.

    Skips (with a printed warning) any key present on only one side or whose
    shape mismatches, rather than raising — a partial transfer is expected
    here, not a bug.
    """
    dst_sd = dst.state_dict()
    matched, skipped = {}, []
    for k, v in src_sd.items():
        if k not in dst_sd:
            skipped.append(f"{tag}.{k} (no such key in fusion_gv module)")
            continue
        if dst_sd[k].shape != v.shape:
            skipped.append(f"{tag}.{k} (shape {tuple(v.shape)} != {tuple(dst_sd[k].shape)})")
            continue
        matched[k] = v
    missing = [k for k in dst_sd if k not in matched]

    dst.load_state_dict(matched, strict=False)

    print(f"[load_vljepa_init] {tag}: loaded {len(matched)}/{len(dst_sd)} tensors")
    for k in skipped:
        print(f"[load_vljepa_init]   skipped src key: {k}")
    for k in missing:
        print(f"[load_vljepa_init]   left at random init: {tag}.{k}")


def load_predictor_and_y_encoder_from_vljepa(model, ckpt_path: str) -> None:
    """Warm-start `model`'s predictor + y_encoder from an open-vljepa checkpoint.

    Args:
        model:     a freshly-constructed FusionGVJEPA (fusion_gv.gvjepa),
                   built with query_model_name != "toy" (a real Llama predictor).
        ckpt_path: path to an open-vljepa checkpoint, e.g. `best.pt` from
                   https://huggingface.co/cun-bjy/open-vljepa (dict with a
                   "model_state_dict" key, or a bare state_dict).
    """
    if not model._use_llama_predictor:
        raise ValueError(
            "load_predictor_and_y_encoder_from_vljepa requires the model to be "
            "built with query_model_name != 'toy' (a real Llama predictor)."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    def _sub(prefix: str) -> dict:
        n = len(prefix)
        return {k[n:]: v for k, v in sd.items() if k.startswith(prefix)}

    _load_matching(model.predictor.layers, _sub("predictor.layers."), "predictor.layers")
    _load_matching(model.predictor.norm, _sub("predictor.norm."), "predictor.norm")
    _load_matching(model.vis_proj, _sub("predictor.vis_proj."), "vis_proj")
    _load_matching(model.pred_proj, _sub("predictor.out_proj."), "pred_proj")
    _load_matching(model.y_encoder, _sub("y_encoder.model."), "y_encoder")
    _load_matching(model.y_proj, _sub("y_encoder.projector."), "y_proj")
