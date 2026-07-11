#!/usr/bin/env python3
"""MMSI-Bench inference for FusionGVJEPA (grounding-head architecture).

Reads the manifest produced by dataset/extract_mmsibench_images.py +
dataset/make_mmsibench_manifest.py. Unlike VSI-Bench, MMSI-Bench rows do NOT
share a fixed image count (2-10 images per row, most rows have 2), so batches
are built by a bucketed sampler (_bucket_batches) that groups same-image-count
rows together and chunks each group by --batch-size independently -- the last
chunk of a bucket just runs at whatever size is left over. Never padded or
mixed across bucket boundaries (collate stacks tensors and requires uniform S
within a batch).

Per sample:
  - text side:  candidate cosine-sim match (via encode_target) -> accuracy,
                broken down by `question_type` and by image count.
  - grounding side: GroundingHead sigmoid heatmap (S, patch_grid, patch_grid).
                Saved as raw heatmap.npy plus per-frame overlay_{k}.jpg.

Input manifest rows:
    {"id": ..., "images": ["img_0.jpg", ...], "num_images": ..., "query": "...",
     "target": "...", "candidates": [...], "question_type": "...", "difficulty": "..."}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fusion_gv.gvjepa_trainer import build_model_from_config
from fusion_gv.infer_gvjepa import _load_checkpoint, _resolve_checkpoint, _score_candidate_batch
from fusion_gv.preprocess import preprocess


def _sanitize_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))


def _save_grounding(image_paths: List[str], heatmap: np.ndarray, out_dir: Path, alpha: float) -> None:
    """Save raw heatmap.npy + per-frame overlay_{k}.jpg (CPU-only, no GPU)."""
    import matplotlib.cm as cm

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "heatmap.npy", heatmap)

    cmap = cm.get_cmap("jet")
    for k, (path, heat) in enumerate(zip(image_paths, heatmap)):
        frame_rgb = Image.open(path).convert("RGB")
        w, h = frame_rgb.size

        heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat_rgba = (cmap(heat_norm) * 255).astype(np.uint8)  # (G, G, 4)
        heat_img = Image.fromarray(heat_rgba, mode="RGBA").resize((w, h), Image.BILINEAR)
        heat_img.putalpha(Image.new("L", (w, h), int(255 * alpha)))

        composite = Image.alpha_composite(frame_rgb.convert("RGBA"), heat_img)
        composite.convert("RGB").save(out_dir / f"overlay_{k}.jpg", quality=90)


class MMSIBenchDataset(Dataset):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)

        # Index-only pass: record byte offsets + num_images per row, never
        # retain parsed rows (see VSIBenchDataset in infer_vsibench.py for
        # why -- avoids O(N) parsed-dict copies across forked DataLoader
        # workers). num_images is needed upfront to build bucket batches.
        self._offsets: list[int] = []
        self.num_images: list[int] = []
        with open(self.manifest_path, "rb") as f:
            offset = f.tell()
            for raw in f:
                if raw.strip():
                    row = json.loads(raw)
                    self._offsets.append(offset)
                    self.num_images.append(int(row.get("num_images") or len(row["images"])))
                offset = f.tell()
        if not self._offsets:
            raise ValueError(f"Empty manifest: {self.manifest_path}")

        self._fh = None

    def __len__(self) -> int:
        return len(self._offsets)

    def _read_row(self, idx: int) -> Dict[str, Any]:
        if self._fh is None:
            self._fh = open(self.manifest_path, "rb")
        self._fh.seek(self._offsets[idx])
        return json.loads(self._fh.readline())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._read_row(idx)
        return {
            "image_paths": row["images"],
            "query": row.get("query", ""),
            "target": row["target"],
            "candidates": row.get("candidates"),
            "id": row.get("id", idx),
            "row_idx": idx,
            "question_type": row.get("question_type"),
            "num_images": len(row["images"]),
        }


def _bucket_batches(num_images: List[int], batch_size: int) -> List[List[int]]:
    """Group row indices by image count, chunk each group by batch_size.

    Does not assume the manifest is pre-sorted by image count (make_mmsibench_
    manifest.py sorts it, but this stays correct either way). The last chunk
    of a bucket is whatever's left over -- never padded, never merged with a
    different image count.
    """
    buckets: Dict[int, List[int]] = {}
    for idx, n in enumerate(num_images):
        buckets.setdefault(n, []).append(idx)

    batches: List[List[int]] = []
    for n in sorted(buckets):
        idxs = buckets[n]
        for i in range(0, len(idxs), batch_size):
            batches.append(idxs[i : i + batch_size])
    return batches


def mmsibench_collate(items: List[Dict[str, Any]], need_vggt: bool) -> Dict[str, Any]:
    vggt_list, jepa_list = [], []
    queries, targets, candidates_list = [], [], []
    ids, row_idxs, qtypes, image_paths_list = [], [], [], []

    for item in items:
        imgs_v, imgs_j = preprocess(item["image_paths"], need_vggt=need_vggt, need_jepa=True)
        if imgs_v is not None:
            vggt_list.append(imgs_v)
        jepa_list.append(imgs_j)
        queries.append(item["query"])
        targets.append(item["target"])
        candidates_list.append(item["candidates"])
        ids.append(item["id"])
        row_idxs.append(item["row_idx"])
        qtypes.append(item["question_type"])
        image_paths_list.append(item["image_paths"])

    images_vggt = torch.cat(vggt_list, dim=0) if vggt_list else None
    images_jepa = torch.cat([j.unsqueeze(0) for j in jepa_list], dim=0).flatten(0, 1)

    return {
        "images_vggt": images_vggt,
        "images_jepa": images_jepa,
        "query": queries,
        "target": targets,
        "candidates": candidates_list,
        "id": ids,
        "row_idx": row_idxs,
        "question_type": qtypes,
        "image_paths": image_paths_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FusionGVJEPA MMSI-Bench inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Max rows per bucket-batch")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--mode", choices=("auto", "select", "multichoices"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None)
    parser.add_argument("--heatmap-alpha", type=float, default=None, help="Overlay opacity (lower = more transparent)")
    parser.add_argument("--no-grounding", action="store_true", help="Skip grounding forward + save entirely")
    parser.add_argument(
        "--render-workers", type=int, default=None,
        help="Thread pool size for grounding overlay rendering (CPU-only), overlapped with the next "
             "batch's GPU forward instead of blocking it. 0 = render synchronously (old behavior).",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    icfg = cfg.get("inference", {})

    manifest_path = Path(args.manifest or icfg.get("manifest", "./data/mmsibench_manifest.jsonl"))
    output_dir = Path(args.output_dir or icfg.get("output_dir", "./outputs/inference/mmsibench"))
    batch_size = args.batch_size or int(icfg.get("batch_size", 8))
    num_workers = args.num_workers if args.num_workers is not None else int(icfg.get("num_workers", 8))
    mode = args.mode or icfg.get("mode", "auto")
    heatmap_alpha = args.heatmap_alpha if args.heatmap_alpha is not None else float(icfg.get("heatmap_alpha", 0.35))
    save_grounding = (not args.no_grounding) and icfg.get("save_grounding", True)
    render_workers = args.render_workers if args.render_workers is not None else int(icfg.get("render_workers", 4))
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    precision = args.precision or icfg.get("precision") or cfg.get("train", {}).get("precision", "bf16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    device_str = args.device or icfg.get("device") or cfg.get("train", {}).get("device", "cuda")
    if device_str != "cpu" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint)
    model = build_model_from_config(cfg)
    if model.grounding_head is None and save_grounding:
        raise RuntimeError(
            "save_grounding=true but config.grounding.enabled is False/missing. "
            "Set `grounding.enabled: true` in the YAML or pass --no-grounding."
        )
    step = _load_checkpoint(model, checkpoint_path)
    model.to(device).eval()

    need_vggt = cfg.get("fusion", {}).get("x_encoder_type", "fusion_gv") == "fusion_gv"

    dataset = MMSIBenchDataset(manifest_path)
    batches = _bucket_batches(dataset.num_images, batch_size)
    bucket_sizes: Dict[int, int] = {}
    for n in dataset.num_images:
        bucket_sizes[n] = bucket_sizes.get(n, 0) + 1
    print(f"Image-count buckets (num_images -> row count): {dict(sorted(bucket_sizes.items()))}")
    print(f"Built {len(batches)} batches (max size {batch_size}, uniform image count per batch)")

    loader = DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=num_workers,
        collate_fn=lambda items: mmsibench_collate(items, need_vggt),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    grounding_dir = output_dir / "grounding"
    if save_grounding:
        grounding_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "mmsibench_predictions.jsonl"
    metrics_path = output_dir / "mmsibench_metrics.json"

    total = 0
    evaluated = 0
    correct = 0
    sim_sum = 0.0
    sim_count = 0
    per_type_total: Dict[str, int] = {}
    per_type_correct: Dict[str, int] = {}
    per_nimg_total: Dict[int, int] = {}
    per_nimg_correct: Dict[int, int] = {}
    autocast_enabled = dtype is not None and device.type == "cuda"

    render_pool = ThreadPoolExecutor(max_workers=render_workers) if render_workers > 0 else None
    pending: List[Future] = []

    with open(predictions_path, "w", encoding="utf-8") as out_f, torch.no_grad():
        for batch in tqdm(loader, desc="mmsibench-infer"):
            images_vggt = batch["images_vggt"].to(device) if batch["images_vggt"] is not None else None
            images_jepa = batch["images_jepa"].to(device)
            queries = batch["query"]
            B = len(queries)
            S = len(batch["image_paths"][0])

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                x_vis, pooled, spatial = model._run_predictor(images_vggt, images_jepa, queries)
                pred_embeddings = model.pred_proj(pooled)

                grounding_heat = None
                if save_grounding:
                    logits = model.grounding_head(x_vis, spatial)          # (B*S, G, G)
                    grounding_heat = logits.sigmoid().float().cpu().numpy().reshape(
                        B, S, logits.shape[-2], logits.shape[-1]
                    )

            candidates_by_row = batch["candidates"]
            should_score = [
                mode in {"select", "multichoices"} or (mode == "auto" and cand)
                for cand in candidates_by_row
            ]
            scoring_candidates = [c if s else [] for c, s in zip(candidates_by_row, should_score)]
            score_results = _score_candidate_batch(
                model, pred_embeddings, scoring_candidates, device, include_scores=False
            )

            sim_scores: List[float | None] = [None] * B
            unscored_idx = [i for i in range(B) if not should_score[i]]
            if unscored_idx:
                unscored_targets = [batch["target"][i] for i in unscored_idx]
                target_emb = model.encode_target(unscored_targets, device)
                pred_sub = torch.nn.functional.normalize(pred_embeddings[unscored_idx].float(), dim=-1)
                target_sub = torch.nn.functional.normalize(target_emb.float(), dim=-1)
                sims = (pred_sub * target_sub).sum(dim=-1)
                for j, i in enumerate(unscored_idx):
                    sim_scores[i] = float(sims[j].item())

            for i in range(B):
                total += 1
                qtype = batch["question_type"][i]
                target_text = batch["target"][i]

                pred_text = None
                is_correct = None
                if should_score[i]:
                    if score_results[i] is None:
                        raise RuntimeError("Missing candidate scores for a row that should be scored.")
                    pred_index, _ = score_results[i]
                    pred_text = candidates_by_row[i][pred_index]
                    is_correct = pred_text == target_text
                    evaluated += 1
                    correct += int(is_correct)
                    if qtype is not None:
                        per_type_total[qtype] = per_type_total.get(qtype, 0) + 1
                        per_type_correct[qtype] = per_type_correct.get(qtype, 0) + int(is_correct)
                    per_nimg_total[S] = per_nimg_total.get(S, 0) + 1
                    per_nimg_correct[S] = per_nimg_correct.get(S, 0) + int(is_correct)

                sim_score = sim_scores[i]
                if sim_score is not None:
                    sim_sum += sim_score
                    sim_count += 1

                grounding_path = None
                if save_grounding:
                    dir_name = f"{batch['row_idx'][i]:06d}_{_sanitize_id(batch['id'][i])}"
                    sample_dir = grounding_dir / dir_name
                    if render_pool is not None:
                        pending.append(render_pool.submit(
                            _save_grounding, batch["image_paths"][i], grounding_heat[i], sample_dir, heatmap_alpha
                        ))
                        if len(pending) > render_workers * 8:
                            still_pending = []
                            for fut in pending:
                                if fut.done():
                                    fut.result()  # re-raise if the render task failed
                                else:
                                    still_pending.append(fut)
                            pending = still_pending
                    else:
                        _save_grounding(batch["image_paths"][i], grounding_heat[i], sample_dir, heatmap_alpha)
                    grounding_path = str(sample_dir.relative_to(output_dir))

                out_f.write(json.dumps({
                    "id": batch["id"][i],
                    "query": queries[i],
                    "question_type": qtype,
                    "num_images": S,
                    "candidates": candidates_by_row[i],
                    "target": target_text,
                    "pred": pred_text,
                    "correct": is_correct,
                    "sim_score": sim_score,
                    "image_paths": batch["image_paths"][i],
                    "grounding_path": grounding_path,
                    "checkpoint": str(checkpoint_path),
                    "step": step,
                }, ensure_ascii=False) + "\n")

    if render_pool is not None:
        for fut in pending:
            fut.result()  # re-raise if any queued render task failed
        render_pool.shutdown(wait=True)

    per_type_accuracy = {
        qtype: per_type_correct[qtype] / per_type_total[qtype] for qtype in per_type_total
    }
    per_nimg_accuracy = {
        n: per_nimg_correct[n] / per_nimg_total[n] for n in per_nimg_total
    }
    metrics = {
        "total": total,
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": (correct / evaluated) if evaluated else None,
        "sim_count": sim_count,
        "mean_sim_score": (sim_sum / sim_count) if sim_count else None,
        "per_question_type": {
            qtype: {"total": per_type_total[qtype], "correct": per_type_correct[qtype], "accuracy": per_type_accuracy[qtype]}
            for qtype in per_type_total
        },
        "per_num_images": {
            str(n): {"total": per_nimg_total[n], "correct": per_nimg_correct[n], "accuracy": per_nimg_accuracy[n]}
            for n in sorted(per_nimg_total)
        },
        "checkpoint": str(checkpoint_path),
        "step": step,
        "mode": mode,
        "manifest": str(manifest_path),
        "save_grounding": save_grounding,
        "heatmap_alpha": heatmap_alpha,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    acc = "n/a" if metrics["accuracy"] is None else f"{metrics['accuracy']:.6f}"
    sim = "n/a" if metrics["mean_sim_score"] is None else f"{metrics['mean_sim_score']:.6f}"
    print(f"wrote predictions: {predictions_path}")
    print(f"wrote metrics: {metrics_path}")
    print(f"evaluated={evaluated} correct={correct} accuracy={acc}")
    print(f"sim_count={sim_count} mean_sim_score={sim}")


if __name__ == "__main__":
    main()
