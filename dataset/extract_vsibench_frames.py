#!/usr/bin/env python3
"""Extract diverse frames from downloaded VSI-Bench videos via Farthest Point Sampling.

Runs once as an offline preprocessing pass so inference never touches
decord/video decode -- it just reads a fixed-count image list per sample,
exactly like fusion_gv training manifests (GVJEPADataset's "images" field).

Frame selection is greedy FPS (as in point-cloud downsampling), not a
fixed-rate/uniform split:
  1. Decode a candidate pool of `--pool-size` frames spread uniformly across
     the clip (cheap: decord only decodes the requested indices).
  2. Reduce each candidate to a small RGB thumbnail feature vector.
  3. Greedily pick --num-frames candidates, each time taking whichever
     remaining candidate has the largest distance to its nearest already-
     picked frame -- so the final set covers the most visually diverse
     moments instead of evenly-spaced (possibly redundant, e.g. static-scene)
     ones. Picks are re-sorted chronologically before saving.

Short clips (fewer raw frames than --num-frames) are padded by repeating the
last frame so every sample ends up with the same S.

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


def _farthest_point_sample(features, num_frames: int) -> List[int]:
    """Greedy FPS over an (N, D) feature array. Returns num_frames indices into it.

    Starts from index 0 (earliest frame) as the anchor, then repeatedly adds
    whichever remaining point maximizes its distance to the nearest already-
    selected point.
    """
    import numpy as np

    n = features.shape[0]
    selected = [0]
    min_dist = np.linalg.norm(features - features[0], axis=1)
    for _ in range(num_frames - 1):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist_to_new = np.linalg.norm(features - features[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist_to_new)
    return selected


def _sample_frame_indices_fps(total_frames: int, num_frames: int, pool_size: int, vr) -> List[int]:
    import numpy as np
    from PIL import Image

    if total_frames <= 0:
        raise RuntimeError("Video has zero frames.")
    if total_frames <= num_frames:
        idx = list(range(total_frames))
        return idx + [idx[-1]] * (num_frames - total_frames)

    pool_n = min(total_frames, max(pool_size, num_frames))
    candidate_idx = np.linspace(0, total_frames - 1, pool_n).astype(int).tolist()
    candidate_frames = vr.get_batch(candidate_idx).asnumpy()  # (pool_n, H, W, C) uint8

    # Small RGB thumbnail per candidate -> cheap FPS distance feature.
    features = np.stack([
        np.asarray(Image.fromarray(f).resize((16, 16), Image.BILINEAR), dtype=np.float32).reshape(-1) / 255.0
        for f in candidate_frames
    ])
    picks = _farthest_point_sample(features, num_frames)
    return sorted(candidate_idx[p] for p in picks)


def _sanitize_id(value) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))


def _extract_one(video_path: str, sample_dir: str, num_frames: int, pool_size: int) -> tuple[str, str | None]:
    """Runs in a worker process. Returns (sample_dir, error_or_None)."""
    from decord import VideoReader, cpu
    from PIL import Image

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        idx = _sample_frame_indices_fps(len(vr), num_frames, pool_size, vr)
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
    parser.add_argument("--out-dir", default=None, help="default: <vsi-dir>/frames{num_frames}_fps")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=64, help="Candidate pool decoded before FPS selection")
    parser.add_argument("--mode", choices=["mc", "open", "all"], default="mc")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    vsi_dir = Path(args.vsi_dir)
    out_dir = Path(args.out_dir or vsi_dir / f"frames{args.num_frames}_fps")
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
            pool.submit(_extract_one, video_path, sample_dir, args.num_frames, args.pool_size): sample_dir
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
