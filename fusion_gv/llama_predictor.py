"""
LlamaPredictor: last-N decoder layers of a pretrained Llama checkpoint, used
non-causally as the VL-JEPA predictor (paper: arXiv 2512.10942, Sec. 3.1).

Also owns the checkpoint's embed_tokens (frozen) so the query-token embedding
lookup and the predictor come from the same loaded weights — one checkpoint
load instead of two.

    full = AutoModelForCausalLM.from_pretrained(llama_name)
        embed_tokens  → kept, frozen   (query-token lookup)
        layers[-N:]   → kept, trainable (the actual predictor)
        norm          → kept, trainable
        rest          → freed
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class LlamaPredictor(nn.Module):
    """
    Args:
        llama_name    : HF checkpoint name (e.g. "meta-llama/Llama-3.2-1B")
        n_keep_layers : number of final decoder layers to keep (trainable)
        hf_cache_dir  : local HF cache dir
        torch_dtype   : dtype to load the checkpoint in (None → fp32)
    """

    def __init__(
        self,
        llama_name: str = "meta-llama/Llama-3.2-1B",
        n_keep_layers: int = 8,
        hf_cache_dir: str | None = None,
        torch_dtype=None,
    ):
        super().__init__()
        full = AutoModelForCausalLM.from_pretrained(
            llama_name, torch_dtype=torch_dtype, cache_dir=hf_cache_dir
        )
        base = full.model  # LlamaModel
        self.hidden_size = base.config.hidden_size

        self.embed_tokens = base.embed_tokens
        self.embed_tokens.requires_grad_(False)   # frozen: query-token lookup table

        self.layers = nn.ModuleList(list(base.layers[-n_keep_layers:]))
        self.norm = base.norm
        self.rotary_emb = getattr(base, "rotary_emb", None)  # RoPE, no learnable params

        # Drop unused references so GC frees the first (N - n_keep_layers) layers,
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
