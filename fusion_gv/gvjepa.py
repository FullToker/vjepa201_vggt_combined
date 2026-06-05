"""
FusionGVJEPA: VL-JEPA training head on top of FusionGV.

Architecture (paper: arXiv 2512.10942v2)
-----------------------------------------
    FusionGV (X-encoder, frozen encoders + concat fusion)
        → list of 4 × (B, S, 1369, D_fused)   D_fused = 3072 for "concat"

    level select + spatial mean-pool
        → (B, S, D_fused)

    vis_proj  → (B, S, D_pred)

    cat [visual tokens | query tokens]
        → Predictor (bidirectional TransformerEncoder)
        → pool over query positions
        → pred_proj  → (B, D_shared)   ← predicted embedding

    Y-Encoder (target text, trainable with slow LR per paper Sec. 3.2)
        → mean-pool token sequence
        → y_proj  → (B, D_shared)      ← target embedding

    Loss: bidirectional InfoNCE(pred, target)

Trainable parameters:
    - fusion module inside FusionGV          (main LR)
    - vis_proj, query_in_proj                (main LR)
    - predictor, pred_proj                   (main LR)
    - y_encoder, y_proj                      (y_encoder_lr_multiplier × main LR)

Frozen parameters:
    - FrozenVGGT, FrozenJEPA (inside FusionGV)
    - query_encoder

Usage
-----
    from fusion_gv.gvjepa import FusionGVJEPA, GVJEPAConfig
    from fusion_gv.config import FusionConfig

    cfg = GVJEPAConfig(fusion=FusionConfig(d_fusion=512))
    model = FusionGVJEPA(cfg).cuda()

    out = model(images_vggt, images_jepa, queries=["..."], targets=["..."])
    # out["pred"]   : (B, D_shared)
    # out["target"] : (B, D_shared)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.nn as nn

from fusion_gv.config import FusionConfig
from fusion_gv.model import build_x_encoder


# ── Toy components for offline / unit-test use ─────────────────────────────────

class _ToyTokenizer:
    """Minimal whitespace tokenizer — not for production quality."""

    def __init__(self, vocab_size: int = 4096) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"

    def _encode(self, text: str, max_length: int) -> list[int]:
        words = text.lower().strip().split()
        ids = [((abs(hash(w)) % (self.vocab_size - 2)) + 2) for w in words]
        ids = ids[: max(1, max_length - 1)]
        ids.append(self.eos_token_id)
        return ids

    def __call__(
        self,
        texts: list[str],
        max_length: int,
        padding: bool,
        truncation: bool,
        return_tensors: str,
    ) -> dict:
        del truncation
        sequences = [self._encode(t, max_length) for t in texts]
        tgt_len = max(len(s) for s in sequences) if padding else None
        padded, mask = [], []
        for seq in sequences:
            if padding and tgt_len is not None:
                pad = tgt_len - len(seq)
                padded.append(seq + [self.pad_token_id] * pad)
                mask.append([1] * len(seq) + [0] * pad)
            else:
                padded.append(seq)
                mask.append([1] * len(seq))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def to(self, device):
        return self


class _ToyTokenBatch(dict):
    def to(self, device):
        return _ToyTokenBatch({k: v.to(device) for k, v in self.items()})


class _ToyTextEncoder(nn.Module):
    """Embedding-only text encoder returning `last_hidden_state`."""

    def __init__(self, vocab_size: int = 4096, hidden_size: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.norm(self.embedding(input_ids)))


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class GVJEPAConfig:
    """Full config for FusionGVJEPA."""

    # Visual fusion encoder
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Predictor (bidirectional transformer)
    predictor_hidden_size: int = 512
    predictor_layers: int = 6
    predictor_heads: int = 8
    predictor_ffn_mult: int = 4
    predictor_dropout: float = 0.0

    # Query encoder (frozen; conditions the predictor)
    # "toy" for offline tests; otherwise a HuggingFace model name
    query_model_name: str = "toy"
    max_query_tokens: int = 64

    # Target (Y) encoder — trainable with slow LR
    y_encoder_name: str = "toy"
    max_target_tokens: int = 64

    # Shared projection dimension (both pred + target projected here)
    shared_embed_dim: int = 512

    # Y-Encoder LR multiplier (paper Sec. 3.2, Tab. 7b)
    y_encoder_lr_multiplier: float = 0.05

    # Which of the 4 fusion levels to use as visual tokens (0=early, 3=final)
    use_fusion_level: int = 3

    # Local directory where HuggingFace model weights are cached
    # All AutoModel / AutoTokenizer downloads land here (mirrors ./ckpts layout)
    hf_cache_dir: str = "./ckpts"


# ── Model ──────────────────────────────────────────────────────────────────────

class FusionGVJEPA(nn.Module):
    """
    FusionGV + VL-JEPA predictor head for vision-(language) pre-training.

    Args:
        config : GVJEPAConfig (uses defaults if None)
    """

    def __init__(self, config: GVJEPAConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = GVJEPAConfig()
        self.config = config

        h = config.predictor_hidden_size
        # Visual dim is configurable so future fusion projections do not require
        # changing this head. Defaults: fusion_gv=3072, vjepa=1024.
        D_fused = config.fusion.visual_dim

        # ── X-encoder (visual) ───────────────────────────────────────────────
        self.x_encoder = build_x_encoder(config.fusion)

        # Project concat-fused spatial features to predictor dim
        self.vis_proj = nn.Linear(D_fused, h)

        # ── Query encoder (frozen) ───────────────────────────────────────────
        if config.query_model_name == "toy":
            self.query_tokenizer = _ToyTokenizer()
            self.query_encoder: nn.Module = _ToyTextEncoder(hidden_size=128)
        else:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError("transformers is required for non-toy query encoder") from exc
            self.query_tokenizer = AutoTokenizer.from_pretrained(
                config.query_model_name,
                use_fast=True,
                cache_dir=config.hf_cache_dir,
            )
            if self.query_tokenizer.pad_token is None:
                self.query_tokenizer.pad_token = self.query_tokenizer.eos_token
            self.query_encoder = AutoModel.from_pretrained(
                config.query_model_name,
                cache_dir=config.hf_cache_dir,
            )

        for p in self.query_encoder.parameters():
            p.requires_grad_(False)

        q_embed_dim = self.query_encoder.get_input_embeddings().embedding_dim
        self.query_in_proj = nn.Linear(q_embed_dim, h)

        # ── Predictor (bidirectional, no causal mask) ────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.predictor_heads,
            dim_feedforward=h * config.predictor_ffn_mult,
            dropout=config.predictor_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(enc_layer, num_layers=config.predictor_layers)
        self.pred_proj = nn.Linear(h, config.shared_embed_dim)

        # ── Y-encoder (target, trainable with slow LR) ───────────────────────
        if config.y_encoder_name == "toy":
            self.y_encoder: nn.Module = _ToyTextEncoder(hidden_size=128)
            self.y_tokenizer = _ToyTokenizer()
        else:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError("transformers is required for non-toy y-encoder") from exc
            self.y_encoder = AutoModel.from_pretrained(
                config.y_encoder_name,
                cache_dir=config.hf_cache_dir,
            )
            self.y_tokenizer = AutoTokenizer.from_pretrained(
                config.y_encoder_name,
                use_fast=True,
                cache_dir=config.hf_cache_dir,
            )
            if self.y_tokenizer.pad_token is None:
                self.y_tokenizer.pad_token = self.y_tokenizer.eos_token

        y_hidden = getattr(self.y_encoder.config, "hidden_size", None)
        if y_hidden is None:
            raise ValueError(f"Y-Encoder '{config.y_encoder_name}' has no hidden_size in config.")
        self.y_proj = nn.Linear(y_hidden, config.shared_embed_dim)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pool_visual(
        self,
        images_vggt: torch.Tensor,
        images_jepa: torch.Tensor,
    ) -> torch.Tensor:
        """Run FusionGV and spatially mean-pool the chosen level.

        Returns: (B, S, D_f)
        """
        feats = self.x_encoder(images_vggt, images_jepa)  # 4 × (B, S, P, D_f)
        feat = feats[self.config.use_fusion_level]         # (B, S, P, D_f)
        return feat.mean(dim=2)                            # (B, S, D_f)

    def _tokenize(self, tokenizer, texts: list[str], max_length: int, device: torch.device):
        out = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in out.items()}

    # ── Public API ─────────────────────────────────────────────────────────────

    def encode_target(self, targets: list[str], device: torch.device) -> torch.Tensor:
        """Y-Encoder branch: target text → (B, D_shared).

        Follows VL-JEPA paper Fig. 1 / Sec. 3.1 (Y-Encoder).
        """
        tok = self._tokenize(self.y_tokenizer, targets, self.config.max_target_tokens, device)
        out = self.y_encoder(**tok)
        hs = out.last_hidden_state                          # (B, L, D_y)
        mask = tok["attention_mask"].unsqueeze(-1).float()
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.y_proj(pooled)                          # (B, D_shared)

    def predict_embedding(
        self,
        images_vggt: torch.Tensor,
        images_jepa: torch.Tensor,
        queries: list[str],
    ) -> torch.Tensor:
        """Predictor branch: (S_V, X_Q) → (B, D_shared).

        Follows VL-JEPA paper Sec. 3.1 (Predictor with bidirectional attention).

        Args:
            images_vggt : (B, S, 3, 518, 518)
            images_jepa : (B*S, 3, 1, 384, 384)
            queries     : list of B query strings (may be empty "")
        """
        device = images_vggt.device
        B = images_vggt.shape[0]

        # Visual stream: FusionGV → spatial pool → project
        vis = self._pool_visual(images_vggt, images_jepa)  # (B, S, D_f)
        vis = self.vis_proj(vis)                           # (B, S, h)
        S = vis.shape[1]

        # Query stream: tokenise → embed → project
        q_tok = self._tokenize(
            self.query_tokenizer, queries, self.config.max_query_tokens, device
        )
        q_emb = self.query_encoder.get_input_embeddings()(q_tok["input_ids"])  # (B, L, q_dim)
        q_emb = self.query_in_proj(q_emb)                                      # (B, L, h)
        q_mask = q_tok["attention_mask"]                                        # (B, L)

        # Concat visual + query tokens, run bidirectional predictor
        x = torch.cat([vis, q_emb], dim=1)                             # (B, S+L, h)
        vis_mask = torch.ones(B, S, device=device, dtype=q_mask.dtype)
        full_mask = torch.cat([vis_mask, q_mask], dim=1)               # (B, S+L)
        x = self.predictor(x, src_key_padding_mask=(full_mask == 0))   # (B, S+L, h)

        # Pool over query positions; fall back to full-sequence mean if query is empty
        q_tokens = x[:, S:, :]                                         # (B, L, h)
        q_mask_f = q_mask.unsqueeze(-1).float()
        pooled_q = (q_tokens * q_mask_f).sum(1) / q_mask_f.sum(1).clamp_min(1)
        fallback = x.mean(dim=1)
        has_q = (q_mask.sum(dim=1) > 0).float().unsqueeze(-1)
        pooled = pooled_q * has_q + fallback * (1.0 - has_q)           # (B, h)

        return self.pred_proj(pooled)                                   # (B, D_shared)

    def forward(
        self,
        images_vggt: torch.Tensor,
        images_jepa: torch.Tensor,
        queries: list[str],
        targets: list[str],
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass returning pred/target embeddings for InfoNCE loss.

        Returns:
            {"pred": (B, D_shared), "target": (B, D_shared)}
        """
        pred = self.predict_embedding(images_vggt, images_jepa, queries)
        target = self.encode_target(targets, images_vggt.device)
        return {"pred": pred, "target": target}

    def parameter_groups(self, lr: float, weight_decay: float) -> List[Dict]:
        """Parameter groups with slow LR for Y-Encoder (paper Sec. 3.2, Tab. 7b)."""
        if lr <= 0:
            raise ValueError("`lr` must be > 0.")
        y_params, base_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("y_encoder."):
                y_params.append(p)
            else:
                base_params.append(p)
        groups = []
        if base_params:
            groups.append({"params": base_params, "lr": lr, "weight_decay": weight_decay})
        if y_params:
            groups.append({
                "params": y_params,
                "lr": lr * self.config.y_encoder_lr_multiplier,
                "weight_decay": weight_decay,
            })
        return groups

    def trainable_parameters(self):
        """Convenience iterator over all trainable parameters."""
        return (p for p in self.parameters() if p.requires_grad)
