#!/usr/bin/env python3
"""Inference and candidate evaluation for FusionGVJEPA.

This reuses the training YAML sections (`fusion` and `model`) to rebuild the
model. Checkpoint resolution order:
    1. --checkpoint
    2. inference.checkpoint
    3. latest step_*.pt in train.output_dir

Input JSONL supports plain embedding inference and SPAR-style candidate eval:
    {"images": [...], "query": "..."}
    {"images": [...], "query": "...", "candidates": [...], "target": "..."}

For candidate rows, the script encodes candidates with the trained Y-encoder,
selects the candidate nearest to the predicted embedding by cosine similarity,
writes per-row predictions, and prints/saves accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fusion_gv.gvjepa_trainer import build_model_from_config
from fusion_gv.preprocess import preprocess


_CANDIDATE_KEYS = ("candidates", "choices", "options", "multichoices")
_TARGET_KEYS = ("target", "answer", "label")


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    cur = cfg
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def _latest_checkpoint(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {output_dir}")

    def step_num(path: Path) -> int:
        match = re.search(r"step_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    candidates = sorted(output_dir.glob("step_*.pt"), key=step_num)
    if not candidates:
        raise FileNotFoundError(f"No step_*.pt checkpoints found in: {output_dir}")
    return candidates[-1]


def _resolve_checkpoint(cfg: dict, override: str | None) -> Path:
    ckpt = override or cfg.get("inference", {}).get("checkpoint")
    if ckpt in {None, "", "null"}:
        train_cfg = cfg.get("train", {})
        if "output_dir" not in train_cfg:
            raise KeyError(
                "No checkpoint specified. Set inference.checkpoint, pass --checkpoint, "
                "or include train.output_dir."
            )
        ckpt = train_cfg["output_dir"]

    path = Path(ckpt)
    if path.is_dir():
        return _latest_checkpoint(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> int | None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    return ckpt.get("step") if isinstance(ckpt, dict) else None


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line"] = line_no
            rows.append(row)
    return rows


def _join_input_root(path: str, input_root: str | Path | None) -> str:
    p = Path(path)
    if p.is_absolute() or input_root in {None, ""}:
        return str(p)
    return str(Path(input_root) / p)


def _image_paths(row: Dict[str, Any], input_root: str | Path | None) -> List[str]:
    if "images" in row:
        paths = row["images"]
    elif "image" in row:
        paths = [row["image"]]
    else:
        raise ValueError(f"Row {row.get('_line', '?')} must contain 'images' or 'image'.")
    if not paths:
        raise ValueError(f"Row {row.get('_line', '?')} has no images.")
    return [_join_input_root(str(p), input_root) for p in paths]


def _raw_image_paths(row: Dict[str, Any]) -> List[str]:
    if "images" in row:
        return list(row["images"])
    if "image" in row:
        return [row["image"]]
    return []


def _candidates(row: Dict[str, Any]) -> List[str]:
    for key in _CANDIDATE_KEYS:
        if key in row:
            values = row[key]
            if isinstance(values, dict):
                values = list(values.values())
            if isinstance(values, str):
                values = [values]
            return [str(v) for v in values]
    return []


def _target(row: Dict[str, Any], candidates: Sequence[str]) -> tuple[str | None, int | None]:
    for key in _TARGET_KEYS:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, int):
            idx = value
            text = candidates[idx] if 0 <= idx < len(candidates) else None
            return text, idx
        text = str(value)
        if text in candidates:
            return text, candidates.index(text)
        return text, None
    return None, None


class _InferenceDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        input_root: str | Path | None,
        need_vggt: bool,
    ) -> None:
        self.rows = rows
        self.input_root = input_root
        self.need_vggt = need_vggt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        imgs_v, imgs_j = preprocess(
            _image_paths(row, self.input_root),
            need_vggt=self.need_vggt,
            need_jepa=True,
        )
        return {
            "row": row,
            "images_vggt": imgs_v,
            "images_jepa": imgs_j,
            "query": row.get("query", ""),
        }


def _infer_collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    vggt_list, jepa_list, queries, rows = [], [], [], []
    expected_s = None
    for item in items:
        imgs_v = item["images_vggt"]
        imgs_j = item["images_jepa"]
        if imgs_j is None:
            raise ValueError("Inference requires V-JEPA inputs.")
        s = imgs_v.shape[1] if imgs_v is not None else imgs_j.shape[0]
        if expected_s is None:
            expected_s = s
        elif s != expected_s:
            raise ValueError(
                "All samples in one inference batch must have the same number of images. "
                f"Got {expected_s} and {s}; set --batch-size 1 for variable-length rows."
            )
        if imgs_v is not None:
            vggt_list.append(imgs_v)
        jepa_list.append(imgs_j)
        queries.append(item["query"])
        rows.append(item["row"])

    if vggt_list:
        images_vggt = torch.cat(vggt_list, dim=0)
    else:
        images_vggt = torch.empty(len(items), expected_s or 0, 0)

    return {
        "rows": rows,
        "images_vggt": images_vggt,
        "images_jepa": torch.cat([j.unsqueeze(0) for j in jepa_list], dim=0).flatten(0, 1),
        "queries": queries,
    }


def _score_candidate_batch(
    model,
    pred_embeddings: torch.Tensor,
    candidates_by_row: List[List[str]],
    device: torch.device,
    include_scores: bool,
) -> List[tuple[int, List[Dict[str, Any]]] | None]:
    flat_candidates: List[str] = []
    spans: List[tuple[int, int] | None] = []
    for candidates in candidates_by_row:
        if not candidates:
            spans.append(None)
            continue
        start = len(flat_candidates)
        flat_candidates.extend(candidates)
        spans.append((start, len(flat_candidates)))

    if not flat_candidates:
        return [None for _ in candidates_by_row]

    target_emb = model.encode_target(flat_candidates, device)
    pred = F.normalize(pred_embeddings.float(), dim=-1)
    target = F.normalize(target_emb.float(), dim=-1)

    results: List[tuple[int, List[Dict[str, Any]]] | None] = []
    for row_idx, span in enumerate(spans):
        if span is None:
            results.append(None)
            continue
        start, end = span
        row_candidates = candidates_by_row[row_idx]
        sims = (pred[row_idx].unsqueeze(0) @ target[start:end].T).squeeze(0)
        order = torch.argsort(sims, descending=True).tolist()
        if include_scores:
            scores = [
                {
                    "index": int(i),
                    "candidate": row_candidates[i],
                    "score": float(sims[i].detach().cpu()),
                }
                for i in order
            ]
        else:
            scores = []
        results.append((int(order[0]), scores))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="FusionGVJEPA inference/evaluation")
    parser.add_argument("--config", required=True, help="Inference or training YAML config")
    parser.add_argument("--input", default=None, help="Input JSONL override")
    parser.add_argument("--output", default=None, help="Prediction JSONL override")
    parser.add_argument("--metrics-output", default=None, help="Metrics JSON override")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file or output directory")
    parser.add_argument("--input-root", default=None, help="Root prepended to relative image paths")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None)
    parser.add_argument("--mode", choices=("auto", "embedding", "select", "multichoices"), default=None)
    parser.add_argument("--x-encoder-type", choices=("fusion_gv", "vjepa"), default=None)
    parser.add_argument("--x-encoder-output-dim", type=int, default=None)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only compute aggregate metrics; do not write per-row predictions.",
    )
    parser.add_argument(
        "--write-scores",
        action="store_true",
        help="When writing predictions, include full per-candidate score lists.",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    icfg = cfg.get("inference", {})

    if args.x_encoder_type is not None:
        _set_nested(cfg, "fusion.x_encoder_type", args.x_encoder_type)
    if args.x_encoder_output_dim is not None:
        _set_nested(cfg, "fusion.x_encoder_output_dim", args.x_encoder_output_dim)

    input_jsonl = Path(args.input or icfg.get("input_jsonl", "./data/spar_eva.jsonl"))
    output_jsonl = Path(args.output or icfg.get("output_jsonl", "./outputs/inference/predictions.jsonl"))
    metrics_output = Path(args.metrics_output or icfg.get("metrics_output", "./outputs/inference/metrics.json"))
    input_root = args.input_root if args.input_root is not None else icfg.get("input_root")
    batch_size = args.batch_size or int(icfg.get("batch_size", 1))
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(icfg.get("num_workers", cfg.get("data", {}).get("num_workers", 16)))
    prefetch_factor = args.prefetch_factor
    if prefetch_factor is None:
        prefetch_factor = int(icfg.get("prefetch_factor", cfg.get("data", {}).get("prefetch_factor", 4)))
    mode = args.mode or icfg.get("mode", "auto")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be > 0")

    precision = args.precision or icfg.get("precision") or cfg.get("train", {}).get("precision", "bf16")
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

    device_str = args.device or icfg.get("device") or cfg.get("train", {}).get("device", "cuda")
    if device_str != "cpu" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint)
    model = build_model_from_config(cfg)
    step = _load_checkpoint(model, checkpoint_path)
    model.to(device).eval()

    x_encoder_type = cfg.get("fusion", {}).get("x_encoder_type", "fusion_gv")
    need_vggt = x_encoder_type == "fusion_gv"

    rows = _read_jsonl(input_jsonl)
    dataset = _InferenceDataset(rows, input_root, need_vggt=need_vggt)
    pin_memory = device.type == "cuda" and not args.no_pin_memory
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _infer_collate,
    }
    if num_workers > 0:
        loader_kwargs.update({
            "prefetch_factor": prefetch_factor,
            "persistent_workers": True,
        })
    loader = DataLoader(dataset, **loader_kwargs)

    if not args.metrics_only:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    evaluated = 0
    correct = 0
    autocast_enabled = dtype is not None and device.type == "cuda"

    out_f = None if args.metrics_only else open(output_jsonl, "w", encoding="utf-8")
    try:
        with torch.no_grad():
            pbar_total = (len(dataset) + batch_size - 1) // batch_size
            for batch in tqdm(loader, total=pbar_total, desc="gvjepa-infer"):
                rows_batch = batch["rows"]
                images_vggt = batch["images_vggt"].to(device, non_blocking=pin_memory)
                images_jepa = batch["images_jepa"].to(device, non_blocking=pin_memory)
                queries = batch["queries"]

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                    pred_embeddings = model.predict_embedding(images_vggt, images_jepa, queries)

                batch_candidates = [_candidates(row) for row in rows_batch]
                batch_targets = [_target(row, candidates) for row, candidates in zip(rows_batch, batch_candidates)]
                batch_should_score = [
                    mode in {"select", "multichoices"} or (mode == "auto" and candidates)
                    for candidates in batch_candidates
                ]
                scoring_candidates = [
                    candidates if should_score else []
                    for candidates, should_score in zip(batch_candidates, batch_should_score)
                ]
                batch_score_results = _score_candidate_batch(
                    model,
                    pred_embeddings,
                    scoring_candidates,
                    device,
                    include_scores=(args.write_scores and not args.metrics_only),
                )

                for row, candidates, target_info, should_score, score_result in zip(
                    rows_batch, batch_candidates, batch_targets, batch_should_score, batch_score_results
                ):
                    total += 1
                    target_text, target_index = target_info

                    if should_score:
                        if not candidates:
                            raise ValueError(f"Row {row.get('_line', '?')} has no candidates for {mode} mode.")
                        if score_result is None:
                            raise RuntimeError("Missing candidate scores for a row that should be scored.")
                        pred_index, scores = score_result
                        pred_text = candidates[pred_index]
                        if target_text is not None:
                            is_correct = pred_text == target_text
                            evaluated += 1
                            correct += int(is_correct)
                    else:
                        pred_index = None
                        pred_text = None
                        scores = []
                        is_correct = None

                    if out_f is not None:
                        result: Dict[str, Any] = {
                            "line": row.get("_line"),
                            "query": row.get("query", ""),
                            "images": _raw_image_paths(row),
                            "checkpoint": str(checkpoint_path),
                        }
                        if step is not None:
                            result["step"] = step
                        if should_score:
                            result.update({
                                "candidates": candidates,
                                "target": target_text,
                                "target_index": target_index,
                                "pred_index": pred_index,
                                "pred": pred_text,
                            })
                            if args.write_scores:
                                result["scores"] = scores
                            if target_text is not None:
                                result["correct"] = is_correct
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()

    metrics = {
        "total": total,
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": (correct / evaluated) if evaluated else None,
        "checkpoint": str(checkpoint_path),
        "step": step,
        "mode": mode,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": None if args.metrics_only else str(output_jsonl),
        "metrics_only": args.metrics_only,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "pin_memory": pin_memory,
        "x_encoder_type": x_encoder_type,
        "need_vggt_preprocess": need_vggt,
    }
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    acc = "n/a" if metrics["accuracy"] is None else f"{metrics['accuracy']:.6f}"
    if args.metrics_only:
        print("metrics-only mode: skipped per-row prediction output")
    else:
        print(f"wrote predictions: {output_jsonl}")
    print(f"wrote metrics: {metrics_output}")
    print(f"evaluated={evaluated} correct={correct} accuracy={acc}")


if __name__ == "__main__":
    main()
