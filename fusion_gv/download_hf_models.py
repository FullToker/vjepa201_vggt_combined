#!/usr/bin/env python3
"""Download HuggingFace models used by FusionGVJEPA into ./ckpts/.

All model files (weights + tokenizer configs) are stored under ./ckpts/ via
the HuggingFace cache_dir mechanism, so they share the same root as the
VGGT and JEPA checkpoints.

Requires /:
    pip install transformers hf
    hf auth login           # for gated models (Llama-3.2)

Usage / :
    python fusion_gv/download_hf_models.py
    python fusion_gv/download_hf_models.py --config fusion_gv/configs/sft_vv_concat.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "ckpts"

DEFAULT_MODELS = {
    "query_encoder": "meta-llama/Llama-3.2-1B",
    "y_encoder": "google/gemma-embedding-300m",
}


def download(model_name: str, cache_dir: Path) -> None:
    from transformers import AutoModel, AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[download] {model_name}  →  {cache_dir}/")

    print("  tokenizer ...")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=str(cache_dir))
    print(f"  tokenizer OK  (vocab_size={tok.vocab_size})")

    print("  model weights ...")
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir))
    hidden = getattr(model.config, "hidden_size", "?")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  model OK  (hidden_size={hidden}, params={params:.0f}M)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config to read model names from")
    args = parser.parse_args()

    models = dict(DEFAULT_MODELS)
    cache_dir = CACHE_DIR

    if args.config:
        import yaml

        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        m = cfg.get("model", {})
        if "query_model_name" in m and m["query_model_name"] != "toy":
            models["query_encoder"] = m["query_model_name"]
        if "y_encoder_name" in m and m["y_encoder_name"] != "toy":
            models["y_encoder"] = m["y_encoder_name"]
        if "hf_cache_dir" in m:
            cache_dir = Path(m["hf_cache_dir"])

    print(f"Cache directory : {cache_dir.resolve()}")
    print(f"Models to fetch : {list(models.values())}")

    for role, name in models.items():
        try:
            download(name, cache_dir)
        except Exception as e:
            print(f"  [FAILED] {role} ({name}): {e}")
            raise

    print("\nAll models downloaded successfully. ")
    print(f"Weights stored under: {cache_dir.resolve()}/")


if __name__ == "__main__":
    main()
