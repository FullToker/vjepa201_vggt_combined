"""
Frozen encoder wrappers.

FrozenVGGT
    Input : (B, S, 3, 518, 518)  float32 [0, 1]
    Output: list of 4 × (B, S, 1369, 2048)
            levels correspond to Aggregator cached rounds {4, 11, 17, 23}

FrozenJEPA
    Input : (B*S, 3, 1, 384, 384)  float32, ImageNet-normalised
    Output: list of 4 × (B, S, 576, 1024)
            levels correspond to ViT-L blocks {5, 11, 17, 23}
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_VGGT_ROOT = _REPO_ROOT / "vggt"
if _LOCAL_VGGT_ROOT.exists() and str(_LOCAL_VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_VGGT_ROOT))

from vggt.models.vggt import VGGT
from app.vjepa_2_1.models import vision_transformer as jepa_vit


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


class FrozenJEPA(nn.Module):
    """
    Loads V-JEPA 2.1 ViT-L encoder from checkpoint and sets out_layers
    so forward() returns intermediate features at 4 levels.
    All parameters are frozen.

    The checkpoint key used is 'ema_encoder' (teacher / EMA weights).

    Output levels (index in output list → ViT-L block):
        0 → block  5  (early semantics)
        1 → block 11  (mid-early semantics)
        2 → block 17  (mid-late semantics)
        3 → block 23  (final semantics)
    """

    _OUT_LAYERS = [5, 11, 17, 23]   # subset of hierarchical_layers for depth=24

    def __init__(self, ckpt_path: str):
        super().__init__()

        self.encoder = jepa_vit.vit_large(
            patch_size=16,
            img_size=(384, 384),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,    # T=1 → image mode
            interpolate_rope=True,
            n_output_distillation=1,
        )

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        enc_state = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state["ema_encoder"].items()
        }
        self.encoder.load_state_dict(enc_state, strict=True)

        # activate multi-level output — forward() returns list instead of tensor
        self.encoder.out_layers = self._OUT_LAYERS

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, B: int, S: int) -> list[torch.Tensor]:
        """
        images : (B*S, 3, 1, 384, 384) float32, ImageNet-normalised
        B, S   : original batch and sequence dimensions
        returns: list of 4 × (B, S, 576, 1024)
        """
        # encoder returns list of 4 × (B*S, 576, 1024) when out_layers is set
        outs = self.encoder(images)
        return [o.view(B, S, 576, 1024) for o in outs]
