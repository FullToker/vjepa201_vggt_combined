#!/usr/bin/env python3
"""Extract FPS-sampled frames from downloaded VSI-Bench videos.

Runs once as an offline preprocessing pass so inference never touches
decord/video decode -- it just reads a fixed-count image list per sample,
exactly like fusion_gv training manifests (GVJEPADataset's "images" field).

Frame indices are picked at a fixed real-time rate (--fps, using each
clip's actual fps, not a proportional/uniform index split) then reduced
(long clips) or padded by repeating the last frame (short clips) to a
fixed --num-frames budget, so every sample ends up with the same S.

Resumable: a video whose frame folder already has --num-frames jpgs is
skipped, so re-running after a partial/interrupted run only fills gaps.

Output layout:
    <out-dir>/<id>/frame_0.jpg .. frame_{num_frames-1}.jpg
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List


def _sample_frame_indices_fps(total_frames: int, video_fps: float, target_fps: float, num_frames: int) -> List[int]:
    if total_frames <= 0:
        raise RuntimeError("Video has zero frames.")
    video_fps = video_fps if video_fps and video_fps > 0 else 30.0
    step = max(video_fps / target_fps, 1.0)

    candidates: List[int] = []
    t = 0.0
    while int(round(t)) < total_frames:
        candidates.append(int(round(t)))
        t += step
    if not candidates:
        candidates = [0]

    if len(candidates) >= num_frames:
        import torch
        pick = torch.linspace(0, len(candidates) - 1, steps=num_frames).long().tolist()
        return [candidates[p] for p in pick]
    return candidates + [candidates[-1]] * (num_frames - len(candidates))


def _sanitize_id(value) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))


def _extract_one(video_path: str, sample_dir: str, fps: float, num_frames: int) -> tuple[str, str | None]:
    """Runs in a worker process. Returns (sample_dir, error_or_None)."""
    from decord import VideoReader, cpu
    from PIL import Image

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        idx = _sample_frame_indices_fps(len(vr), vr.get_avg_fps(), fps, num_frames)
        frames = vr.get_batch(idx).asnumpy()  # (T, H, W, C) uint8
    except Exception as e:  # noqa: BLE001 - report and move on, don't kill the whole run
        return sample_dir, f"{type(e).__name__}: {e}"

    out = Path(sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, frame in enumerate(frames):
        Image.fromarray(frame).save(out / f"frame_{k}.jpg", quality=95)
    return sample_dir, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi-dir", default="./source_data/vsibench")
    parser.add_argument("--out-dir", default=None, help="default: <vsi-dir>/frames")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--mode", choices=["mc", "open", "all"], default="mc")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    vsi_dir = Path(args.vsi_dir)
    out_dir = Path(args.out_dir or vsi_dir / "frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = vsi_dir / "test.jsonl"

    with open(src, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    jobs = []
    skipped_missing_video = already_done = 0
    for row in rows:
        has_options = bool(row.get("options"))
        if args.mode == "mc" and not has_options:
            continue
        if args.mode == "open" and has_options:
            continue

        video_path = vsi_dir / row["dataset"] / f"{row['scene_name']}.mp4"
        if not video_path.exists():
            skipped_missing_video += 1
            continue

        sample_dir = out_dir / _sanitize_id(row["id"])
        if len(list(sample_dir.glob("frame_*.jpg"))) == args.num_frames:
            already_done += 1
            continue

        jobs.append((str(video_path), str(sample_dir)))

    print(f"Total rows: {len(rows)} | already extracted: {already_done} | "
          f"missing video: {skipped_missing_video} | to extract: {len(jobs)}")

    extracted = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_extract_one, video_path, sample_dir, args.fps, args.num_frames): sample_dir
            for video_path, sample_dir in jobs
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            sample_dir, err = fut.result()
            if err is not None:
                failed += 1
                print(f"  FAILED {sample_dir}: {err}")
            else:
                extracted += 1
            if i % 100 == 0:
                print(f"  progress: {i}/{len(jobs)}")

    print(f"Done. Extracted: {extracted}, Failed: {failed}")


if __name__ == "__main__":
    main()
