#!/usr/bin/env python3
"""Convert EmbodiedScan PKL + VG JSON → FusionGVJEPA grounding JSONL.

Directory layout (--data-root):
  <data_root>/
    embodiedscan/embodiedscan/
      embodiedscan_infos_{train,val,test}.pkl
      embodiedscan_{train,val,test}_vg.json
    scannet/
      posed_images/
        sceneXXXX_XX/
          XXXXX.jpg

Each output row:
  {
    "images":    ["abs/path/view0.jpg", ...],   # max_views visible views
    "query":     "the board beside the door",
    "target":    "the board beside the door",
    "task_type": "grounding",
    "boxes":     [[x1,y1,x2,y2]|null, ...],    # per-view, 518x518 pixel space
    "source":    "embodiedscan",
  }
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np

_VGGT_SIZE = 518


def bbox7_to_corners(bbox7: list | np.ndarray) -> np.ndarray:
    """[cx,cy,cz, l,w,h, yaw] → (8,3) world-frame corners."""
    cx, cy, cz = float(bbox7[0]), float(bbox7[1]), float(bbox7[2])
    l, w, h    = float(bbox7[3]), float(bbox7[4]), float(bbox7[5])
    yaw        = float(bbox7[6])

    signs = np.array([
        [-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
        [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1],
    ], dtype=np.float64)
    local = signs * np.array([l / 2, w / 2, h / 2])

    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return local @ Rz.T + np.array([cx, cy, cz])


def project_box(
    corners_w: np.ndarray,
    cam2global: np.ndarray,
    cam2img: np.ndarray,
    orig_hw: tuple[int, int],
    axis_align_matrix: np.ndarray | None = None,
) -> list[float] | None:
    """Project 8 world corners → [x1,y1,x2,y2] in 518×518 space, or None.

    corners_w:          (8,3) in axis-aligned world frame (bbox_3d frame).
    cam2global:         (4,4) camera-to-original-world (mmdet3d convention).
    axis_align_matrix:  (4,4) aligned = axis_align_matrix @ original → need inv to go back.
    cam2img:            (4,4) or (3,3) intrinsic — only top-left 3×3 used.
    """
    H, W = orig_hw

    # corners_w is in axis-aligned frame; cam2global is in original (unaligned) frame.
    # Convert corners back to original world frame via inv(axis_align_matrix).
    if axis_align_matrix is not None:
        align_inv = np.linalg.inv(axis_align_matrix)
        ones = np.ones((corners_w.shape[0], 1))
        corners_h = np.hstack([corners_w, ones])        # (8,4)
        corners_w = (align_inv @ corners_h.T).T[:, :3]  # (8,3) original world frame

    R = cam2global[:3, :3]
    t = cam2global[:3, 3]
    cam_pts = (corners_w - t) @ R          # (8,3)  [R.T via right-multiply]

    valid = cam_pts[:, 2] > 0
    if not valid.any():
        return None

    pts = cam_pts[valid]                   # (K,3)
    K   = cam2img[:3, :3]
    uvw = pts @ K.T
    u   = uvw[:, 0] / uvw[:, 2]
    v   = uvw[:, 1] / uvw[:, 2]

    scale      = _VGGT_SIZE / min(H, W)
    new_H      = round(H * scale)
    new_W      = round(W * scale)
    u          = u * scale - (new_W - _VGGT_SIZE) / 2.0
    v          = v * scale - (new_H - _VGGT_SIZE) / 2.0
    u          = np.clip(u, 0., _VGGT_SIZE)
    v          = np.clip(v, 0., _VGGT_SIZE)

    x1, y1, x2, y2 = float(u.min()), float(v.min()), float(u.max()), float(v.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def infer_hw(cam2img: np.ndarray) -> tuple[int, int]:
    """Estimate (H, W) from principal point (cx, cy) in cam2img."""
    cx, cy = float(cam2img[0, 2]), float(cam2img[1, 2])
    return round(cy * 2), round(cx * 2)


def fps_sample(views: list[dict], k: int) -> list[dict]:
    """Farthest Point Sampling on camera positions (cam2global[:3, 3]).

    Greedily selects k views maximising minimum pairwise distance in 3D
    camera-position space.  Seed is index 0 (deterministic).
    """
    if len(views) <= k:
        return views
    positions = np.array([np.asarray(v["cam2global"], dtype=np.float64)[:3, 3]
                          for v in views])  # (N, 3)
    N = len(positions)
    selected: list[int] = [0]
    min_dists = np.full(N, np.inf)
    min_dists[0] = 0.0
    for _ in range(k - 1):
        last = positions[selected[-1]]
        dists = np.sum((positions - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1.0
        selected.append(int(np.argmax(min_dists)))
    return [views[i] for i in selected]


def uniform_sample(views: list[dict], k: int) -> list[dict]:
    if len(views) <= k:
        return views
    step = len(views) / k
    return [views[int(i * step)] for i in range(k)]


def sample_view_groups(
    pool: list[dict],
    max_views: int,
    num_groups: int,
    sampling: str,
) -> list[list[dict]]:
    """Partition pool into up to `num_groups` disjoint view groups of size `max_views`.

    Each group is drawn (without replacement across groups) from the shrinking
    remainder, so the same view never appears twice for one VG entry. Stops
    early if the pool runs out before `num_groups` groups are formed.
    """
    sample_fn = fps_sample if sampling == "fps" else uniform_sample
    remaining = list(pool)
    groups: list[list[dict]] = []
    for _ in range(num_groups):
        if not remaining:
            break
        selected = sample_fn(remaining, max_views)
        groups.append(selected)
        selected_ids = {id(v) for v in selected}
        remaining = [v for v in remaining if id(v) not in selected_ids]
    return groups


def convert(
    split: str,
    data_root: str | Path,
    output_jsonl: str | Path,
    max_views: int = 8,
    sampling: str = "fps",
    num_groups: int = 4,
) -> int:
    data_root    = Path(data_root)
    ann_dir      = data_root / "embodiedscan" / "embodiedscan"
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Load PKL and build scan_id → sample lookup
    pkl_path = ann_dir / f"embodiedscan_infos_{split}.pkl"
    data     = pickle.load(open(pkl_path, "rb"))
    scan2sample: dict[str, dict] = {
        s["sample_idx"]: s
        for s in data["data_list"]
        if s["sample_idx"].startswith("scannet/") and "cam2img" in s
    }

    # Load VG JSON (use _all variant when available for full annotations)
    vg_path_all = ann_dir / f"embodiedscan_{split}_vg_all.json"
    vg_path     = vg_path_all if vg_path_all.exists() else ann_dir / f"embodiedscan_{split}_vg.json"
    print(f"Loading VG from {vg_path}")
    vg_entries = json.load(open(vg_path))

    rows_written      = 0
    skipped_no_sample = 0
    skipped_no_inst   = 0
    skipped_invisible = 0

    with open(output_jsonl, "w") as out:
        for entry in vg_entries:
            scan_id   = entry["scan_id"]      # e.g. "scannet/scene0191_00"
            target_id = entry["target_id"]
            text      = entry["text"]

            sample = scan2sample.get(scan_id)
            if sample is None:
                skipped_no_sample += 1
                continue

            # Find target instance by bbox_id
            inst = next(
                (i for i in sample["instances"] if i["bbox_id"] == target_id),
                None,
            )
            if inst is None:
                skipped_no_inst += 1
                continue

            corners_w         = bbox7_to_corners(inst["bbox_3d"])
            cam2img           = np.asarray(sample["cam2img"], dtype=np.float64)
            axis_align_matrix = np.asarray(sample["axis_align_matrix"], dtype=np.float64)
            orig_hw           = infer_hw(cam2img)

            # Prefer views where target is visible; uniformly sample for diversity
            all_views = sample["images"]
            visible   = [
                v for v in all_views
                if target_id in v.get("visible_instance_ids", [])
            ]
            pool   = visible if visible else all_views
            groups = sample_view_groups(pool, max_views, num_groups, sampling)

            wrote_any = False
            for selected in groups:
                images: list[str]               = []
                boxes:  list[list[float] | None] = []

                for v in selected:
                    cam2global = np.asarray(v["cam2global"], dtype=np.float64)
                    box        = project_box(corners_w, cam2global, cam2img, orig_hw, axis_align_matrix)
                    images.append(str(data_root / v["img_path"]))
                    boxes.append(box)

                if not any(b is not None for b in boxes):
                    continue

                out.write(json.dumps({
                    "images":    images,
                    "query":     text,
                    "target":    text,
                    "task_type": "grounding",
                    "boxes":     boxes,
                    "source":    "embodiedscan",
                }) + "\n")
                rows_written += 1
                wrote_any = True

            if not wrote_any:
                skipped_invisible += 1

    print(
        f"[{split}] wrote {rows_written} rows  "
        f"(skipped: {skipped_no_sample} no-sample, "
        f"{skipped_no_inst} no-inst, {skipped_invisible} invisible)"
    )
    return rows_written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert EmbodiedScan PKL+VG → FusionGVJEPA grounding JSONL."
    )
    p.add_argument("--split",     required=True, choices=["train", "val", "test"],
                   help="Dataset split")
    p.add_argument("--data-root", required=True,
                   help="Parent dir containing embodiedscan/ and scannet/")
    p.add_argument("--output",    required=True, help="Output .jsonl path")
    p.add_argument("--max-views", type=int, default=8,
                   help="Max camera views per grounding entry (default: 8)")
    p.add_argument("--sampling", choices=["fps", "uniform"], default="fps",
                   help="View selection strategy: fps (default) or uniform")
    p.add_argument("--num-groups", type=int, default=4,
                   help="Max disjoint view-groups (rows) per VG entry (default: 4)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert(args.split, args.data_root, args.output, args.max_views, args.sampling, args.num_groups)
