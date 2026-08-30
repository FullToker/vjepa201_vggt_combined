"""
LlamaPredictor: last-N decoder layers of a pretrained Llama checkpoint, used
non-causally as the VL-JEPA predictor (paper: arXiv 2512.10942, Sec. 3.1).

Also owns the checkpoint's embed_tokens (frozen) so the query-token embedding
lookup and the predictor come from the same loaded weights — one checkpoint
load instead of two.

    full = AutoModelForCausalLM.from_pretrained(llama_name)
        embed_tokens        → kept, frozen    (query-token lookup)
        layers[:n_query]    → kept, trainable (optional query self-attn, off by default)
        layers[-n_keep:]    → kept, trainable (the actual predictor)
        norm                → kept, trainable
        rest                → freed

Optional query self-attention (n_query_layers > 0, default 0 = current
behavior, unchanged):
    The paper's default keeps only the tail n_keep_layers of a 16-layer Llama
    -- the first (16 - n_keep_layers) layers are loaded then immediately
    discarded, unused. Query tokens reach the shared predictor as raw,
    zero-context embed_tokens lookups (no self-attention among themselves
    before fusing with visual tokens) -- the predictor has to both "read" the
    question AND fuse AND reason, in the same n_keep_layers.

    n_query_layers > 0 reuses that many of the otherwise-discarded early
    layers (base.layers[:n_query_layers]) as a small CAUSAL pre-encoder for
    the query alone (see forward_query below), run BEFORE concatenation with
    visual tokens -- giving the query its own contextualization stage using
    real pretrained weights instead of a from-scratch module. Causal (not
    bidirectional like the main predictor) because that's the masking these
    layers were pretrained under. Default 0 keeps every existing config's
    behavior bit-for-bit reproducible; this is an architectural addition made
    in this repo, not something the paper validated, and n_query_layers +
    n_keep_layers must not exceed the checkpoint's layer count (raises).
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class LlamaPredictor(nn.Module):
    """
    Args:
        llama_name      : HF checkpoint name (e.g. "meta-llama/Llama-3.2-1B")
        n_keep_layers   : number of final decoder layers to keep (trainable)
        n_query_layers  : number of INITIAL decoder layers to additionally
                          keep as a causal query pre-encoder (see module
                          docstring). 0 (default) = off, current behavior.
        hf_cache_dir    : local HF cache dir
        torch_dtype     : dtype to load the checkpoint in (None → fp32)
    """

    def __init__(
        self,
        llama_name: str = "meta-llama/Llama-3.2-1B",
        n_keep_layers: int = 8,
        n_query_layers: int = 0,
        hf_cache_dir: str | None = None,
        torch_dtype=None,
    ):
        super().__init__()
        full = AutoModelForCausalLM.from_pretrained(
            llama_name, torch_dtype=torch_dtype, cache_dir=hf_cache_dir
        )
        base = full.model  # LlamaModel
        self.hidden_size = base.config.hidden_size
        self.n_query_layers = n_query_layers

        n_total = len(base.layers)
        if n_query_layers + n_keep_layers > n_total:
            raise ValueError(
                f"n_query_layers({n_query_layers}) + n_keep_layers({n_keep_layers}) "
                f"= {n_query_layers + n_keep_layers} exceeds {llama_name}'s "
                f"{n_total} layers -- they must not overlap."
            )

        self.embed_tokens = base.embed_tokens
        self.embed_tokens.requires_grad_(False)   # frozen: query-token lookup table

        # Front slice (query pre-encoder, optional) and tail slice (predictor)
        # come from disjoint, non-overlapping layer ranges of the same checkpoint.
        self.query_layers = (
            nn.ModuleList(list(base.layers[:n_query_layers])) if n_query_layers > 0 else None
        )
        self.layers = nn.ModuleList(list(base.layers[-n_keep_layers:]))
        self.norm = base.norm
        self.rotary_emb = getattr(base, "rotary_emb", None)  # RoPE, no learnable params

        # Drop unused references so GC frees the middle (unused) layers,
        # lm_head, etc.
        base.layers = nn.ModuleList()
        base.embed_tokens = None
        base.norm = None
        if hasattr(base, "rotary_emb"):
            base.rotary_emb = None
        del full

    def get_input_embeddings(self) -> nn.Embedding:
        """Frozen token-embedding table, HF-style accessor."""
        return self.embed_tokens

    @staticmethod
    def _build_bidirectional_mask(
        attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build a non-causal additive attention mask from a 2D pad mask.

        Args:
            attention_mask: (B, L) — 1 for valid, 0 for pad
            dtype: target dtype matching hidden states
        Returns:
            (B, 1, L, L) additive mask — 0 for attend, -inf for pad column
        """
        B, L = attention_mask.shape
        pad = 1.0 - attention_mask.to(torch.float32)          # 1 where pad
        neg_inf = torch.finfo(dtype).min
        mask = pad[:, None, None, :] * neg_inf                 # (B,1,1,L)
        return mask.expand(B, 1, L, L).to(dtype)

    @staticmethod
    def _build_causal_mask(
        attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build a causal + pad additive attention mask -- the masking these
        layers were pretrained under, used for forward_query (unlike the
        main predictor's forward, which is deliberately non-causal).

        Args:
            attention_mask: (B, L) — 1 for valid, 0 for pad
            dtype: target dtype matching hidden states
        Returns:
            (B, 1, L, L) additive mask — 0 where attend allowed, -inf elsewhere
        """
        B, L = attention_mask.shape
        neg_inf = torch.finfo(dtype).min
        causal = torch.triu(
            torch.full((L, L), neg_inf, device=attention_mask.device), diagonal=1
        )                                                       # (L,L)
        pad = (1.0 - attention_mask.to(torch.float32)) * neg_inf  # (B,L)
        mask = causal[None, None, :, :] + pad[:, None, None, :]   # (B,1,L,L)
        return mask.to(dtype)

    def forward_query(
        self,
        x: torch.Tensor,                 # (B, L, hidden_size)
        attention_mask: torch.Tensor,    # (B, L) — 1 for valid, 0 for pad
    ) -> torch.Tensor:
        """Causal self-attention pre-encoder for query tokens only, using
        query_layers (base.layers[:n_query_layers]). No-op (returns x
        unchanged) when n_query_layers == 0. Natural word-order positions
        (0..L-1) -- this runs before concatenation with visual tokens, so it
        has no modality/frame offsets to account for yet. Does NOT apply
        self.norm (that's the main predictor's final norm, applied once at
        the end of forward() below, not here).
        """
        if self.query_layers is None:
            return x

        B, L = x.shape[:2]
        position_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        attn_mask_4d = self._build_causal_mask(attention_mask, x.dtype)

        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(x, position_ids)

        for layer in self.query_layers:
            layer_out = layer(
                x,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        return x

    def forward(
        self,
        x: torch.Tensor,                 # (B, L, hidden_size)
        position_ids: torch.Tensor,      # (L,) or (B, L)
        attention_mask: torch.Tensor,    # (B, L) — 1 for valid, 0 for pad
    ) -> torch.Tensor:
        B = x.shape[0]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(B, -1)

        attn_mask_4d = self._build_bidirectional_mask(attention_mask, x.dtype)

        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(x, position_ids)

        for layer in self.layers:
            layer_out = layer(
                x,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        return self.norm(x)
