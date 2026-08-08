#!/usr/bin/env python3
"""Zero-shot (no SFT) VSI-Bench / MMSI-Bench candidate-eval, using open-vljepa's
own pretrained checkpoint and model classes (../open-vljepa) directly.

Not fusion_gv/FusionGVJEPA -- this is the raw open-vljepa architecture,
unmodified. Purpose: a zero-training baseline number to compare against the
warm-started+SFT'd fusion_gv runs (sft_from_vljepa_ckpt*.yaml). If SFT barely
beats this, the gap is in the SFT setup (data/domain/training), not in
whether warm-starting from open-vljepa is a good idea in principle.

V-JEPA2 (open-vljepa's X-encoder) is a video model and already accepts a
variable number of frames per sample natively (RoPE, no code changes needed)
-- see openvljepa/models/encoder.py::VJEPA2Encoder. This script just feeds it
each manifest row's images list as-is.

Manifest schema (same as fusion_gv/infer_vsibench.py / infer_mmsibench.py):
    {"id": ..., "images": ["frame_0.jpg", ...], "query": "...", "target": "...",
     "question_type": "...", "candidates": [...]}   # candidates optional

Usage:
    python openvl/infer_bench.py --config openvl/configs/zeroshot_infer.yaml --config-section vsi_inference
    python openvl/infer_bench.py --config openvl/configs/zeroshot_infer.yaml --config-section mmsi_inference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# openvl/infer_bench.py -> openvl/ -> vjepa2/ -> Program/ -> Program/open-vljepa
_DEFAULT_OPENVLJEPA_ROOT = Path(__file__).resolve().parents[2] / "open-vljepa"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _add_openvljepa_to_path(repo_root: Path) -> None:
    if not repo_root.exists():
        raise FileNotFoundError(
            f"open-vljepa repo not found at {repo_root}. Pass --openvljepa-repo "
            f"if your checkout isn't a sibling directory of this project."
        )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_frames(image_paths: List[str], image_size: int) -> torch.Tensor:
    """Matches openvljepa.data.msrvtt.MSRVTTDataset's transform exactly
    (Resize short-side -> CenterCrop -> ImageNet normalize), so the checkpoint
    sees the same input distribution it was trained on.

    Returns: (T, 3, image_size, image_size) float32.
    """
    transform = T.Compose([
        T.Resize(image_size, antialias=True),
        T.CenterCrop(image_size),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    frames = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        frames.append(arr)
    return transform(torch.stack(frames))


class BenchDataset(Dataset):
    """Reads the same manifest schema as fusion_gv's infer_vsibench.py /
    infer_mmsibench.py. Whole file loaded in memory (VSI/MMSI manifests are
    small, unlike SPAR's multi-million-row ones -- no need for the
    offset-indexed lazy read those use)."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.rows: List[Dict[str, Any]] = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest_path}")
        self.num_images = [len(r["images"]) for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {
            "image_paths": row["images"],
            "query": row.get("query", ""),
            "target": row["target"],
            "candidates": row.get("candidates"),
            "question_type": row.get("question_type"),
            "id": row.get("id", idx),
        }


def _bucket_batches(num_images: List[int], batch_size: int) -> List[List[int]]:
    """Group row indices so every batch has a uniform image count (V-JEPA2
    needs matching T to stack a batch) -- mirrors fusion_gv/infer_mmsibench.py's
    bucketing for MMSI-Bench's variable per-row image count (2-10). VSI-Bench
    manifests have a fixed count, so this degenerates to plain contiguous
    batching there."""
    buckets: Dict[int, List[int]] = {}
    for i, n in enumerate(num_images):
        buckets.setdefault(n, []).append(i)
    batches: List[List[int]] = []
    for idxs in buckets.values():
        for s in range(0, len(idxs), batch_size):
            batches.append(idxs[s : s + batch_size])
    return batches


def make_collate(image_size: int, query_tokenizer, max_query_len: int):
    def collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack(
            [load_frames(item["image_paths"], image_size) for item in items]
        )  # (B, T, 3, H, W)

        qenc = query_tokenizer(
            [item["query"] or "Describe the scene." for item in items],
            max_length=max_query_len, padding="max_length", truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "query_ids": qenc["input_ids"],
            "query_mask": qenc["attention_mask"],
            "target": [item["target"] for item in items],
            "candidates": [item["candidates"] for item in items],
            "question_type": [item["question_type"] for item in items],
            "id": [item["id"] for item in items],
        }

    return collate


@torch.no_grad()
def score_candidates(
    model,
    target_tokenizer,
    pred: torch.Tensor,
    candidates_by_row: List[Optional[List[str]]],
    max_target_len: int,
    device: torch.device,
) -> List[Optional[int]]:
    """Cosine-sim nearest candidate per row -- mirrors
    fusion_gv/infer_gvjepa.py's _score_candidate_batch, against openvljepa's
    own YEncoder instead of FusionGVJEPA.encode_target."""
    flat: List[str] = []
    spans: List[Optional[tuple[int, int]]] = []
    for cands in candidates_by_row:
        if not cands:
            spans.append(None)
            continue
        start = len(flat)
        flat.extend(cands)
        spans.append((start, len(flat)))
    if not flat:
        return [None] * len(candidates_by_row)

    tenc = target_tokenizer(
        flat, max_length=max_target_len, padding="max_length", truncation=True,
        return_tensors="pt",
    )
    target_emb = model.y_encoder(
        tenc["input_ids"].to(device), tenc["attention_mask"].to(device)
    )
    pred_n = F.normalize(pred.float(), dim=-1)
    target_n = F.normalize(target_emb.float(), dim=-1)

    results: List[Optional[int]] = []
    for row_idx, span in enumerate(spans):
        if span is None:
            results.append(None)
            continue
        start, end = span
        sims = pred_n[row_idx] @ target_n[start:end].T
        results.append(int(sims.argmax().item()))
    return results


def run_eval(
    model, query_tokenizer, target_tokenizer,
    cfg: dict, icfg: dict, device: torch.device, dtype,
) -> None:
    manifest_path = icfg["manifest"]
    output_dir = Path(icfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(icfg.get("batch_size", 8))
    num_workers = int(icfg.get("num_workers", 8))
    image_size = int(cfg.get("image_size", 256))
    max_query_len = int(cfg.get("max_query_len", 64))
    max_target_len = int(cfg.get("max_target_len", 64))

    dataset = BenchDataset(manifest_path)
    batches = _bucket_batches(dataset.num_images, batch_size)
    print(
        f"[{manifest_path}] {len(dataset)} rows -> {len(batches)} batches "
        f"(uniform image count per batch)"
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=num_workers,
        collate_fn=make_collate(image_size, query_tokenizer, max_query_len),
    )

    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

    total = evaluated = correct = 0
    per_type_total: Dict[str, int] = {}
    per_type_correct: Dict[str, int] = {}
    autocast_enabled = dtype is not None and device.type == "cuda"

    with open(predictions_path, "w", encoding="utf-8") as out_f:
        for batch in tqdm(loader, desc=f"openvl-zeroshot[{output_dir.name}]"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            query_ids = batch["query_ids"].to(device, non_blocking=True)
            query_mask = batch["query_mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                pred = model(pixel_values, query_ids, query_mask)  # target_ids=None -> returns pred (B, D)

            pred_idx = score_candidates(
                model, target_tokenizer, pred, batch["candidates"], max_target_len, device
            )

            B = pixel_values.shape[0]
            for i in range(B):
                total += 1
                cands = batch["candidates"][i]
                target_text = batch["target"][i]
                qtype = batch["question_type"][i]

                is_correct = None
                pred_text = None
                if cands and pred_idx[i] is not None:
                    pred_text = cands[pred_idx[i]]
                    is_correct = pred_text == target_text
                    evaluated += 1
                    correct += int(is_correct)
                    if qtype is not None:
                        per_type_total[qtype] = per_type_total.get(qtype, 0) + 1
                        per_type_correct[qtype] = per_type_correct.get(qtype, 0) + int(is_correct)

                out_f.write(json.dumps({
                    "id": batch["id"][i],
                    "question_type": qtype,
                    "target": target_text,
                    "pred": pred_text,
                    "correct": is_correct,
                }, ensure_ascii=False) + "\n")

    metrics = {
        "total": total,
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": (correct / evaluated) if evaluated else None,
        "per_type": {
            t: {
                "total": per_type_total[t],
                "correct": per_type_correct.get(t, 0),
                "accuracy": per_type_correct.get(t, 0) / per_type_total[t],
            }
            for t in per_type_total
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"wrote predictions: {predictions_path}")
    print(f"wrote metrics: {metrics_path}")
    print(f"evaluated={evaluated} correct={correct} accuracy={metrics['accuracy']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot open-vljepa candidate-eval on VSI-Bench/MMSI-Bench."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--config-section", required=True,
        help="Top-level YAML key to read this run's manifest/output_dir/batch_size "
             "from (e.g. 'vsi_inference', 'mmsi_inference').",
    )
    parser.add_argument(
        "--openvljepa-repo", default=None,
        help="Path to the open-vljepa repo checkout. Default: ../open-vljepa "
             "relative to this project (sibling directory).",
    )
    args = parser.parse_args()

    repo_root = Path(args.openvljepa_repo) if args.openvljepa_repo else _DEFAULT_OPENVLJEPA_ROOT
    _add_openvljepa_to_path(repo_root)

    from transformers import AutoTokenizer

    from openvljepa.data.msrvtt import _ensure_pad_token
    from openvljepa.models.vljepa import OpenVLJEPA

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    icfg = cfg.get(args.config_section, {})
    if not icfg:
        raise KeyError(f"Config section '{args.config_section}' missing/empty in {args.config}")

    device_str = icfg.get("device") or "cuda"
    if device_str != "cpu" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    precision = icfg.get("precision", "bf16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

    ckpt_path = cfg["checkpoint"]
    print(f"Loading open-vljepa checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]

    model = OpenVLJEPA(
        model_cfg["encoder"], model_cfg["y_encoder"], model_cfg["predictor"],
        torch_dtype=dtype,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    non_xencoder_missing = [k for k in missing if not k.startswith("x_encoder.model.")]
    if non_xencoder_missing:
        print(f"  WARNING: missing non-frozen keys: {non_xencoder_missing[:5]}...")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected[:5]}...")
    print(f"  Loaded {len(ckpt['model_state_dict'])} keys (epoch={ckpt.get('epoch', '?')})")

    model.eval().to(device)

    query_tokenizer = _ensure_pad_token(
        AutoTokenizer.from_pretrained(model_cfg["predictor"]["llama_name"])
    )
    target_tokenizer = _ensure_pad_token(
        AutoTokenizer.from_pretrained(model_cfg["y_encoder"]["model_name"])
    )

    run_eval(model, query_tokenizer, target_tokenizer, cfg, icfg, device, dtype)


if __name__ == "__main__":
    main()
