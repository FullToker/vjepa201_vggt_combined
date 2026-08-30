"""
FusionGVJEPA: VL-JEPA training head on top of FusionGV.

Architecture (paper: arXiv 2512.10942v2)
-----------------------------------------
    FusionGV (X-encoder, frozen encoders + final-level fusion)
        → (B, S, 1369, D_fused)   D_fused = 2 * proj_dim (default 2048)

    VisualResampler (K learnable queries cross-attend over each frame's
    patch tokens — Perceiver-style, content-adaptive; generalizes mean-pool)
        → (B, S*K, D_fused)       K = config.visual_pool_k (hyperparameter)

    vis_proj  → (B, S*K, D_pred)
        + modality_embed[visual] + frame_embed(frame_ids)   (same frame_ids
          row repeated across a frame's K tokens; no order/distance)

    query_in_proj(embed_tokens(query_ids))  → (B, L, D_pred)
        → [optional] causal query pre-encoder: query_self_attn_layers INITIAL
          Llama layers (otherwise-discarded, disjoint from predictor_llama_layers'
          tail slice), causal, natural word-order positions. 0 (default) = off.
        + modality_embed[query] (+ query_pos_embed sincos — toy path only;
                                    LlamaPredictor's RoPE covers word order)

    cat [visual tokens | query tokens]
        → Predictor:
            - query_model_name == "toy": bidirectional nn.TransformerEncoder
            - otherwise: LlamaPredictor — last predictor_llama_layers decoder
              layers of query_model_name, non-causal, RoPE.
              position_ids = [0]*S + arange(1, L+1)  (all visual tokens share
              one position → zero relative RoPE rotation among frames; query
              tokens get normal sequential positions)
        → pool over query positions
        → pred_proj  → (B, D_shared)   ← predicted embedding

    Y-Encoder (target text, trainable with slow LR per paper Sec. 3.2)
        → mean-pool token sequence
        → y_proj  → (B, D_shared)      ← target embedding

    Loss: bidirectional InfoNCE(pred, target)

Trainable parameters:
    - fusion module inside FusionGV          (main LR)
    - visual_resampler                       (main LR)
    - vis_proj, query_in_proj                (main LR)
    - modality_embed, frame_embed            (main LR)
    - predictor, pred_proj                   (main LR)
      (LlamaPredictor: only the kept decoder layers (query_layers +
       predictor's own tail layers, when query_self_attn_layers > 0) + final
       norm are trainable; its embed_tokens stays frozen)
    - y_encoder, y_proj                      (y_encoder_lr_multiplier × main LR)

Frozen parameters:
    - FrozenVGGT, FrozenJEPA (inside FusionGV)
    - query_encoder (toy path) / predictor.embed_tokens (llama path)

Usage
-----
    from fusion_gv.gvjepa import FusionGVJEPA, GVJEPAConfig
    from fusion_gv.config import FusionConfig

    cfg = GVJEPAConfig(fusion=FusionConfig(proj_dim=1024))
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

from app.vjepa_2_1.models.utils.pos_embs import get_1d_sincos_pos_embed
from fusion_gv.config import FusionConfig
from fusion_gv.grounding_head import CrossAttnLayer, GroundingHead
from fusion_gv.llama_predictor import LlamaPredictor
from fusion_gv.model import build_x_encoder

# Max frames a single forward pass can tag with frame_embed (index range [0, _S_MAX)).
_S_MAX = 33


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


# ── Visual resampler ─────────────────────────────────────────────────────────

class VisualResampler(nn.Module):
    """Perceiver-style learnable pooling: K learnable queries cross-attend
    over each frame's patch tokens, replacing spatial mean-pool with
    content-adaptive pooling. K is a hyperparameter (config.visual_pool_k)
    — the predictor's self-attention cost scales with (num_frames*K + L)^2,
    so K trades spatial fidelity against predictor compute.

    Reuses CrossAttnLayer from grounding_head.py (same cross-attention
    building block, just refining K learned queries instead of a
    predictor-conditioned query).
    """

    def __init__(
        self,
        dim: int,
        k: int,
        num_layers: int = 1,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert k >= 1, "visual_pool_k must be >= 1"
        self.k = k
        self.query = nn.Parameter(torch.randn(k, dim) * dim**-0.5)
        self.layers = nn.ModuleList(
            [CrossAttnLayer(dim, num_heads, ffn_mult, dropout) for _ in range(num_layers)]
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (N, P, dim) → (N, K, dim), N = B * num_frames."""
        q = self.query.unsqueeze(0).expand(patches.shape[0], -1, -1)
        for layer in self.layers:
            q, _ = layer(q, patches)
        return q


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class GVJEPAConfig:
    """Full config for FusionGVJEPA."""

    # Visual fusion encoder
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Visual resampler (replaces spatial mean-pool): K learnable queries
    # cross-attend over each frame's patch tokens (Perceiver-style,
    # content-adaptive pooling). K is the main compute/fidelity knob —
    # predictor self-attention cost is O((num_frames*K + L)^2). K=1 keeps
    # today's token budget but is a *learned* single-token pool, not a
    # literal average — old mean-pool checkpoints do not transfer to
    # visual_resampler's weights and it needs (re)training.
    visual_pool_k: int = 1
    visual_pool_layers: int = 1
    visual_pool_heads: int = 8
    visual_pool_ffn_mult: int = 4
    visual_pool_dropout: float = 0.0

    # Predictor. When query_model_name == "toy": a small bidirectional
    # nn.TransformerEncoder sized by the fields below (fast, no download).
    # Otherwise: LlamaPredictor — last predictor_llama_layers decoder layers
    # of query_model_name, non-causal, RoPE. hidden size is then derived from
    # the checkpoint (2048 for Llama-3.2-1B), predictor_hidden_size is ignored.
    predictor_hidden_size: int = 512    # toy path only
    predictor_layers: int = 6           # toy path only
    predictor_heads: int = 8            # toy path only
    predictor_ffn_mult: int = 4         # toy path only
    predictor_dropout: float = 0.0      # toy path only
    predictor_llama_layers: int = 8     # llama path only
    # Optional causal query pre-encoder: this many of query_model_name's
    # INITIAL layers (llama path only), otherwise-unused/discarded, run the
    # query through self-attention before it's concatenated with visual
    # tokens -- see fusion_gv/llama_predictor.py's module docstring. 0
    # (default) = off, current behavior unchanged. Must satisfy
    # query_self_attn_layers + predictor_llama_layers <= checkpoint's layer
    # count (LlamaPredictor raises otherwise).
    query_self_attn_layers: int = 0     # llama path only

    # Query encoder (frozen; conditions the predictor). "toy" for offline
    # tests; otherwise a HuggingFace Llama checkpoint name — also used as the
    # predictor backbone (see predictor_llama_layers above).
    query_model_name: str = "toy"
    max_query_tokens: int = 64

    # Target (Y) encoder — trainable with slow LR
    y_encoder_name: str = "toy"
    max_target_tokens: int = 64

    # Shared projection dimension (both pred + target projected here)
    shared_embed_dim: int = 512

    # Y-Encoder LR multiplier (paper Sec. 3.2, Tab. 7b)
    y_encoder_lr_multiplier: float = 0.05

    # Local directory where HuggingFace model weights are cached
    # All AutoModel / AutoTokenizer downloads land here (mirrors ./ckpts layout)
    hf_cache_dir: str = "./ckpts"

    # Grounding head (optional, for EmbodiedScan bbox supervision)
    grounding_enabled: bool = False
    grounding_num_layers: int = 3
    grounding_num_heads: int = 8
    grounding_ffn_mult: int = 4
    grounding_dropout: float = 0.0
    grounding_patch_grid: int = 37   # 518 / 14 = 37


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

        # Visual dim is configurable so future fusion projections do not require
        # changing this head. Defaults: fusion_gv=2048, vjepa=1024.
        D_fused = config.fusion.visual_dim

        # ── X-encoder (visual) ───────────────────────────────────────────────
        self.x_encoder = build_x_encoder(config.fusion)

        # ── Visual pooling ───────────────────────────────────────────────────
        # k == 1 -> pure spatial mean-pool, zero params. This is the original
        # (pre-VisualResampler) behaviour, kept so mean-pool checkpoints trained
        # before commit 89714e7 still load strict=True. k > 1 -> learnable
        # Perceiver-style resampler.
        if config.visual_pool_k > 1:
            self.visual_resampler: VisualResampler | None = VisualResampler(
                dim=D_fused,
                k=config.visual_pool_k,
                num_layers=config.visual_pool_layers,
                num_heads=config.visual_pool_heads,
                ffn_mult=config.visual_pool_ffn_mult,
                dropout=config.visual_pool_dropout,
            )
        else:
            self.visual_resampler = None

        # ── Predictor + query tokenizer ───────────────────────────────────────
        self._use_llama_predictor = config.query_model_name != "toy"

        if self._use_llama_predictor:
            from transformers import AutoTokenizer

            self.query_tokenizer = AutoTokenizer.from_pretrained(
                config.query_model_name,
                use_fast=True,
                cache_dir=config.hf_cache_dir,
            )
            if self.query_tokenizer.pad_token is None:
                self.query_tokenizer.pad_token = self.query_tokenizer.eos_token

            # Last-8-layer Llama predictor. Also owns the frozen embed_tokens
            # used for query-token lookup (one checkpoint load, not two).
            self.predictor = LlamaPredictor(
                llama_name=config.query_model_name,
                n_keep_layers=config.predictor_llama_layers,
                n_query_layers=config.query_self_attn_layers,
                hf_cache_dir=config.hf_cache_dir,
            )
            h = self.predictor.hidden_size   # forced by the checkpoint (2048)
            self.query_in_proj = nn.Identity()
        else:
            self.query_tokenizer = _ToyTokenizer()
            self.query_encoder: nn.Module = _ToyTextEncoder(hidden_size=128)
            for p in self.query_encoder.parameters():
                p.requires_grad_(False)

            h = config.predictor_hidden_size
            q_embed_dim = self.query_encoder.get_input_embeddings().embedding_dim
            self.query_in_proj = nn.Linear(q_embed_dim, h)

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

        # Project fused spatial features to predictor dim
        self.vis_proj = nn.Linear(D_fused, h)

        # ── Embeddings added before the predictor ─────────────────────────────
        # modality_embed: 0=visual, 1=query — lets the predictor tell the two
        # streams apart (it otherwise has no positional signal at all).
        self.modality_embed = nn.Embedding(2, h)
        # frame_embed: per-frame group tag. Same row broadcast over every one
        # of a frame's K tokens (K = config.visual_pool_k, via VisualResampler
        # — see frame_ids in _run_predictor). No order/distance semantics —
        # frame sampling here carries no temporal meaning, so this is a pure
        # "same-group" marker, not a position encoding.
        self.frame_embed = nn.Embedding(_S_MAX, h)
        # query_pos_embed: sincos, word order for query tokens. Only needed on
        # the toy path — LlamaPredictor's own RoPE (recomputed every layer)
        # already encodes query word order, so it's redundant there.
        if not self._use_llama_predictor:
            query_pos_np = get_1d_sincos_pos_embed(h, config.max_query_tokens)
            self.register_buffer(
                "query_pos_embed", torch.from_numpy(query_pos_np).float(), persistent=False
            )

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

        # ── Grounding head (optional) ─────────────────────────────────────────
        self.grounding_head: GroundingHead | None = None
        if config.grounding_enabled:
            self.grounding_head = GroundingHead(
                spatial_dim=D_fused,
                hidden_dim=h,
                num_layers=config.grounding_num_layers,
                num_heads=config.grounding_num_heads,
                ffn_mult=config.grounding_ffn_mult,
                dropout=config.grounding_dropout,
                patch_grid=config.grounding_patch_grid,
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pool_visual(
        self,
        images_vggt: torch.Tensor | None,
        images_jepa: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the X-encoder and pool its final-level output via the
        learnable VisualResampler (K tokens/frame, frame-major layout).

        Args:
            images_vggt: may be None under x_encoder_type="vjepa" when the
                caller skipped building it (e.g. infer_vsibench.py's
                need_vggt optimization).
            batch_size: B, needed to split images_jepa's flattened (B*S, ...)
                shape back into (B, S) when images_vggt (which otherwise
                carries both numbers) is None.

        Returns:
            pooled:  (B, S*K, D_f)     K resampled tokens/frame, frame-major
                                        (frame0_tok0..K-1, frame1_tok0..K-1, ...)
            spatial: (B, S, P, D_f)    raw patch features before pooling
        """
        with torch.profiler.record_function("x_encoder"):
            feat = self.x_encoder(images_vggt, images_jepa, batch_size)   # (B, S, P, D_f)
        B, S, P, D_f = feat.shape
        with torch.profiler.record_function("visual_resampler"):
            if self.visual_resampler is None:
                pooled = feat.mean(dim=2)                                     # (B, S, D_f)  K=1 mean-pool
            else:
                pooled = self.visual_resampler(feat.reshape(B * S, P, D_f))   # (B*S, K, D_f)
                pooled = pooled.reshape(B, S * self.visual_resampler.k, D_f)  # frame-major
        return pooled, feat                                 # (B,S*K,D_f), (B,S,P,D_f)

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

    def _run_predictor(
        self,
        images_vggt: torch.Tensor | None,
        images_jepa: torch.Tensor,
        queries: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared predictor forward used by both InfoNCE and grounding paths.

        Args:
            images_vggt: may be None under x_encoder_type="vjepa" (see
                _pool_visual) -- device/B are then taken from images_jepa/queries.

        Returns:
            x_vis:   (B, S, h)      predictor output over visual positions (summary)
            pooled:  (B, h)         query-pooled embedding (for pred_proj)
            spatial: (B, S, P, D_f) raw spatial features before pooling
        """
        device = images_vggt.device if images_vggt is not None else images_jepa.device
        B = len(queries)

        pooled_vis, spatial = self._pool_visual(images_vggt, images_jepa, B)
        vis = self.vis_proj(pooled_vis)   # (B, S, h), S = num_frames * K
        S = vis.shape[1]
        S_frames = spatial.shape[1]
        assert S_frames <= _S_MAX, f"frame_embed only covers {_S_MAX} frames, got S={S_frames}"

        # Same frame_ids row repeated across each frame's K tokens (frame-major
        # layout from _pool_visual) — a pure "same-group" marker, not a K-index.
        frame_ids = torch.arange(S_frames, device=device).repeat_interleave(S // S_frames)
        vis = vis + self.modality_embed.weight[0] + self.frame_embed(frame_ids)  # (B,S,h)

        q_tok = self._tokenize(
            self.query_tokenizer, queries, self.config.max_query_tokens, device
        )
        if self._use_llama_predictor:
            q_emb = self.predictor.get_input_embeddings()(q_tok["input_ids"])  # (B, L, h), frozen
        else:
            q_emb = self.query_encoder.get_input_embeddings()(q_tok["input_ids"])  # (B, L, q_dim)
        q_emb = self.query_in_proj(q_emb)                                      # (B, L, h)
        q_mask = q_tok["attention_mask"]                                        # (B, L)

        # Optional causal query pre-encoder (llama path only, no-op when
        # query_self_attn_layers=0) — runs BEFORE modality_embed is added, so
        # these layers see plain query hidden states, same as during their
        # original Llama pretraining. See llama_predictor.py.
        if self._use_llama_predictor:
            q_emb = self.predictor.forward_query(q_emb, q_mask)                # (B, L, h)

        L = q_emb.shape[1]
        q_emb = q_emb + self.modality_embed.weight[1]                          # (B,L,h)
        if not self._use_llama_predictor:
            q_emb = q_emb + self.query_pos_embed[:L]

        x = torch.cat([vis, q_emb], dim=1)                              # (B, S+L, h)
        vis_mask = torch.ones(B, S, device=device, dtype=q_mask.dtype)
        full_mask = torch.cat([vis_mask, q_mask], dim=1)                # (B, S+L), 1=valid

        with torch.profiler.record_function("predictor_forward"):
            if self._use_llama_predictor:
                position_ids = torch.cat([
                    torch.zeros(S, dtype=torch.long, device=device),
                    torch.arange(1, L + 1, dtype=torch.long, device=device),
                ])
                x = self.predictor(x, position_ids=position_ids, attention_mask=full_mask)
            else:
                x = self.predictor(x, src_key_padding_mask=(full_mask == 0))    # (B, S+L, h)

        x_vis = x[:, :S, :]                                             # (B, S, h)

        q_tokens = x[:, S:, :]                                          # (B, L, h)
        q_mask_f = q_mask.unsqueeze(-1).float()
        pooled_q = (q_tokens * q_mask_f).sum(1) / q_mask_f.sum(1).clamp_min(1)
        fallback = x.mean(dim=1)
        has_q = (q_mask.sum(dim=1) > 0).float().unsqueeze(-1)
        pooled = pooled_q * has_q + fallback * (1.0 - has_q)            # (B, h)

        return x_vis, pooled, spatial

    def predict_embedding(
        self,
        images_vggt: torch.Tensor,
        images_jepa: torch.Tensor,
        queries: list[str],
    ) -> torch.Tensor:
        """Predictor branch: (S_V, X_Q) → (B, D_shared).

        Args:
            images_vggt : (B, S, 3, 518, 518)
            images_jepa : (B*S, 3, 1, 384, 384)
            queries     : list of B query strings (may be empty "")
        """
        _, pooled, _ = self._run_predictor(images_vggt, images_jepa, queries)
        return self.pred_proj(pooled)                                    # (B, D_shared)

    def forward_grounding(
        self,
        images_vggt: torch.Tensor,
        images_jepa: torch.Tensor,
        queries: list[str],
    ) -> torch.Tensor:
        """Grounding forward: returns spatial attention logits for seg supervision.

        Requires grounding_enabled=True in config.

        Returns:
            logits: (B*S, patch_grid, patch_grid)  raw pre-softmax attention logits
                    Apply sigmoid + BCE against gt_mask derived from projected 2D bbox.
        """
        if self.grounding_head is None:
            raise RuntimeError("grounding_enabled=False; set it in GVJEPAConfig.")
        x_vis, _, spatial = self._run_predictor(images_vggt, images_jepa, queries)
        return self.grounding_head(x_vis, spatial)                       # (B*S, G, G)

    def forward(
        self,
        images_vggt: torch.Tensor,
        images_jepa: torch.Tensor,
        queries: list[str],
        targets: list[str] | None = None,
        mode: str = "infonce",
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass — single entry point so DDP always sees the same
        callable regardless of which branch (InfoNCE vs grounding) runs.

        Args:
            mode: "infonce" (default) returns pred/target embeddings.
                  "grounding" returns spatial attention logits instead
                  (requires grounding_enabled=True); `targets` is ignored.

        Returns:
            mode="infonce":   {"pred": (B, D_shared), "target": (B, D_shared),
                                "x_vis": (B, S, h), "spatial": (B, S, P, D_f)}
            mode="grounding": {"grounding_logits": (B*S, patch_grid, patch_grid)}
        """
        x_vis, pooled, spatial = self._run_predictor(images_vggt, images_jepa, queries)

        if mode == "grounding":
            if self.grounding_head is None:
                raise RuntimeError("grounding_enabled=False; set it in GVJEPAConfig.")
            logits = self.grounding_head(x_vis, spatial)
            return {"grounding_logits": logits}

        pred = self.pred_proj(pooled)
        target = self.encode_target(targets, images_vggt.device)
        return {"pred": pred, "target": target, "x_vis": x_vis, "spatial": spatial}

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
