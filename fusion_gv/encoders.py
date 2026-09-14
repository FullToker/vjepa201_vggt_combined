"""
Frozen encoder wrappers.

FrozenVGGT
    Input : (B, S, 3, 518, 518)  float32 [0, 1]
    Output: list of 4 × (B, S, 1369, 2048)
            levels correspond to Aggregator cached rounds {4, 11, 17, 23}

FrozenSemanticEncoder
    Generic wrapper driven by an encoder_registry.EncoderSpec -- e.g.
        FrozenSemanticEncoder(SEMANTIC_ENCODERS["vjepa"], ckpt_path)
        FrozenSemanticEncoder(SEMANTIC_ENCODERS["ijepa"], ckpt_path)
    Input : (B*S, 3, [1,] spec.img_size, spec.img_size)  float32, ImageNet-normalised
            (the T=1 dim is present only when spec.add_temporal_dim is True)
    Output: list of len(spec.out_layers) × (B, S, spec.num_patches, spec.embed_dim)

    All parameters frozen. See encoder_registry.py to add a new spec instead
    of adding a new class here.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

from fusion_gv.encoder_registry import EncoderSpec

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_VGGT_ROOT = _REPO_ROOT / "vggt"
if _LOCAL_VGGT_ROOT.exists() and str(_LOCAL_VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_VGGT_ROOT))

from vggt.models.vggt import VGGT


class FrozenVGGT(nn.Module):
    """
    Loads the full VGGT-1B checkpoint and exposes only the Aggregator.
    All parameters are frozen.

    Output levels (index in output list → Aggregator round):
        0 → round  4  (early geometry)
        1 → round 11  (mid-early geometry)
        2 → round 17  (mid-late geometry)
        3 → round 23  (final geometry)
    """

    _PATCH_START_IDX = 5   # 1 camera token + 4 register tokens

    def __init__(self, ckpt_path: str):
        super().__init__()

        vggt = VGGT()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        vggt.load_state_dict(state)

        self.aggregator = vggt.aggregator

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """
        images : (B, S, 3, 518, 518) float32 [0, 1]
        returns: list of 4 × (B, S, 1369, 2048)
        """
        token_list, patch_start_idx = self.aggregator(images)
        # token_list: 24 elements, None for non-cached rounds
        # non-None shape: (B, S, P, 2C)  P = 1 + 4 + 1369 = 1374
        feats = [
            t[:, :, patch_start_idx:, :]   # strip camera + register → (B, S, 1369, 2048)
            for t in token_list
            if t is not None
        ]
        return feats   # list of 4


class FrozenSemanticEncoder(nn.Module):
    """
    Generic frozen semantic-encoder wrapper, parameterized by an
    encoder_registry.EncoderSpec. Replaces what used to be a separate
    FrozenJEPA / FrozenIJEPA class per encoder -- adding a new encoder means
    adding a spec to encoder_registry.SEMANTIC_ENCODERS, not a new class here.
    """

    def __init__(self, spec: EncoderSpec, ckpt_path: str):
        super().__init__()
        self.spec = spec
        self.encoder = spec.build_backbone()

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        enc_state = state[spec.ckpt_state_key]
        for prefix in spec.strip_prefixes:
            enc_state = {k.replace(prefix, ""): v for k, v in enc_state.items()}
        self.encoder.load_state_dict(enc_state, strict=True)

        # activate multi-level output — forward() returns list instead of tensor
        self.encoder.out_layers = list(spec.out_layers)

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, B: int, S: int) -> list[torch.Tensor]:
        """
        images : (B*S, 3, [1,] spec.img_size, spec.img_size) float32, ImageNet-normalised
        B, S   : original batch and sequence dimensions
        returns: list of len(spec.out_layers) × (B, S, spec.num_patches, spec.embed_dim)
        """
        # encoder returns list of N × (B*S, num_patches, embed_dim) when out_layers is set
        outs = self.encoder(images)
        return [o.view(B, S, self.spec.num_patches, self.spec.embed_dim) for o in outs]
