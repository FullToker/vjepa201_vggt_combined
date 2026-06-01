# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"

ARCH_NAME_MAP = {
    # V-JEPA 2.1
    "vjepa2_1_vit_base_384": ("vit_base", "vjepa2_1_vitb_dist_vitG_384"),
    "vjepa2_1_vit_large_384": ("vit_large", "vjepa2_1_vitl_dist_vitG_384"),
    "vjepa2_1_vit_giant_384": ("vit_giant_xformers", "vjepa2_1_vitg_384"),
    "vjepa2_1_vit_gigantic_384": ("vit_gigantic_xformers", "vjepa2_1_vitG_384"),
}


def _clean_backbone_key(state_dict):
    for key, val in state_dict.copy().items():
        _ = state_dict.pop(key)
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        state_dict[key] = val
    return state_dict


# ########## V-JEPA 2.1 ##########


vjepa2_1_teacher_embed_dim = 1664


def _make_vjepa2_1_model(
    model_name: str = "vjepa2_1_vit_large_384",
    checkpoint_key="target_encoder",
    img_size=384,
    patch_size=16,
    tubelet_size=2,
    num_frames=64,
    predictor_embed_dim=384,
    predictor_depth=24,
    predictor_num_mask_tokens=10,
    n_output_distillation=4,
    return_all_tokens=False,
    teacher_embed_dim=None,
    pretrained: bool = True,
    **kwargs,
):
    from app.vjepa_2_1.models import predictor as vit_predictor, vision_transformer as vit_encoder

    vit_encoder_kwargs = dict(
        patch_size=patch_size,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    vit_encoder_kwargs.update(**kwargs)

    arch_name = ARCH_NAME_MAP[model_name][0]
    encoder = vit_encoder.__dict__[arch_name](**vit_encoder_kwargs)

    vit_predictor_kwargs = dict(
        img_size=(img_size, img_size),
        patch_size=patch_size,
        use_mask_tokens=True,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        teacher_embed_dim=teacher_embed_dim,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        depth=predictor_depth,
        num_heads=12,
        num_mask_tokens=predictor_num_mask_tokens,
        use_rope=True,
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        n_output_distillation=n_output_distillation,
        return_all_tokens=return_all_tokens,
        img_temporal_dim_size=1,
    )
    vit_predictor_kwargs.update(**kwargs)

    predictor = vit_predictor.__dict__["vit_predictor"](**vit_predictor_kwargs)

    if pretrained:
        model_file = ARCH_NAME_MAP[model_name][-1]
        url = VJEPA_BASE_URL + f"/{model_file}.pt"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        encoder_state_dict = _clean_backbone_key(state_dict[checkpoint_key])
        encoder.load_state_dict(
            encoder_state_dict, strict=True
        )  # state_dict has pos_embed but we use RoPE
        predictor_state_dict = _clean_backbone_key(state_dict["predictor"])
        predictor.load_state_dict(
            predictor_state_dict, strict=True
        )  # state_dict has pos_embed but we use RoPE

    return encoder, predictor


def vjepa2_1_vit_base_384(*, pretrained: bool = True, **kwargs):
    return _make_vjepa2_1_model(
        model_name="vjepa2_1_vit_base_384",
        checkpoint_key="ema_encoder",
        img_size=384,
        predictor_depth=12,
        predictor_num_mask_tokens=8,
        n_output_distillation=1,
        return_all_tokens=True,
        teacher_embed_dim=vjepa2_1_teacher_embed_dim,
        pretrained=pretrained,
        **kwargs,
    )


def vjepa2_1_vit_large_384(*, pretrained: bool = True, **kwargs):
    return _make_vjepa2_1_model(
        model_name="vjepa2_1_vit_large_384",
        checkpoint_key="ema_encoder",
        img_size=384,
        predictor_depth=12,
        predictor_num_mask_tokens=8,
        n_output_distillation=1,
        return_all_tokens=True,
        teacher_embed_dim=vjepa2_1_teacher_embed_dim,
        pretrained=pretrained,
        **kwargs,
    )


def vjepa2_1_vit_giant_384(*, pretrained: bool = True, **kwargs):
    return _make_vjepa2_1_model(
        model_name="vjepa2_1_vit_giant_384",
        img_size=384,
        predictor_num_mask_tokens=8,
        n_output_distillation=4,
        return_all_tokens=True,
        pretrained=pretrained,
        **kwargs,
    )


def vjepa2_1_vit_gigantic_384(*, pretrained: bool = True, **kwargs):
    return _make_vjepa2_1_model(
        model_name="vjepa2_1_vit_gigantic_384",
        img_size=384,
        predictor_num_mask_tokens=8,
        n_output_distillation=4,
        return_all_tokens=True,
        pretrained=pretrained,
        **kwargs,
    )
