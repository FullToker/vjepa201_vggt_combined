#!/usr/bin/env python3
"""Build a fusion_gv inference manifest from OpenEQA's open-eqa-v0.json.

Selects --num-frames per question via greedy Farthest Point Sampling (FPS)
on camera position, not uniform time sampling -- picks frames that spread
out spatially across the episode instead of evenly-spaced (possibly
redundant, e.g. dwelling in one room) ones.

Two episode sources, two pose readers:

  hm3d-v0/*    -- dataset/download_openeqa.sh output. Camera pose is
                   reconstructed per-frame from the raw agent_state pickle
                   (<frames-dir>/<episode>/*.pkl, from OpenEQA's
                   open-eqa-hm3d-states-v0.tgz) via the fixed sensor offset
                   hardcoded in the upstream repo's data/hm3d/config.py
                   (sensor 1.0m above agent origin, zero pitch) -- NOT via
                   the upstream extract-frames.py's save_pose(), which
                   routes through a live habitat_sim.Simulator and therefore
                   requires the gated/token-walled HM3D scene mesh. Reading
                   agent_state.position/.rotation directly needs the
                   `habitat_sim` package importable (for the AgentState
                   class) but never touches the mesh or GPU.
                   CAVEAT: the sensor-offset math here has not been
                   cross-checked against upstream's own pose.txt output --
                   spot-check one episode before trusting this at scale.

  scannet-v0/* -- reuses this repo's existing ScanNet posed-image data
                   (dataset/download_scannet_posed.sh output,
                   <scannet-dir>/posed_images/<scene_id>/<frame>.{jpg,txt})
                   instead of OpenEQA's own gated ScanNet extraction path.
                   Pose lives right next to each frame as a plain-text 4x4
                   camera-to-world matrix (standard ScanNet posed-image
                   convention) -- no external annotation pkl needed. Invalid
                   frames (ScanNet marks bad tracking as all -inf) are
                   dropped before FPS. Skipped entirely (not fatal) if
                   <scannet-dir>/posed_images/<scene_id>/ isn't found.

Output manifest rows (fusion_gv GVJEPADataset-compatible; no `candidates`
field -- OpenEQA is open-ended only, so infer_openeqa.py scores rows via
prediction<->target cosine similarity, not multiple-choice matching):
    {"id": ..., "images": ["frame_0-rgb.png", ...], "query": "...",
     "target": "...", "question_type": "<category>"}
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _fps_indices(positions: np.ndarray, k: int) -> list[int]:
    """Greedy FPS over an (N, 3) position array. Returns k indices, seeded at 0."""
    n = positions.shape[0]
    if n <= k:
        return list(range(n))
    selected = [0]
    min_dist = np.linalg.norm(positions - positions[0], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        d = np.linalg.norm(positions - positions[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
    return selected


def _pad(images: list[str], num_frames: int) -> list[str]:
    return images + [images[-1]] * (num_frames - len(images)) if images else images


# ---------------------------------------------------------------------------
# hm3d-v0: pose from raw agent_state pickle (no Simulator, no scene mesh)
# ---------------------------------------------------------------------------

_SENSOR_LOCAL_OFFSET = np.array([0.0, 1.0, 0.0])  # data/hm3d/config.py: sensor_position height-only


def _quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _hm3d_camera_position(pkl_path: Path) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)  # needs `habitat_sim` importable to unpickle agent_state
    st = data["agent_state"]
    R = _quat_to_matrix(st.rotation)
    pos = np.asarray(st.position, dtype=np.float64)
    return pos + R @ _SENSOR_LOCAL_OFFSET


def hm3d_pose_reader_available(frames_dir: Path, episode_history: str) -> bool:
    """One-time probe: can we unpickle a state file in this episode at all?"""
    folder = frames_dir / episode_history
    probe = next(iter(sorted(folder.glob("*.pkl"))), None)
    if probe is None:
        return False
    try:
        _hm3d_camera_position(probe)
        return True
    except ModuleNotFoundError as e:
        print(
            f"NOTE: can't read HM3D pose pickles ({e}) -- "
            "`pip install habitat-sim` (or conda) to unpickle agent_state. "
            "All hm3d-v0/* rows will be skipped."
        )
        return False


def sample_hm3d_episode(frames_dir: Path, episode_history: str, num_frames: int) -> list[str] | None:
    folder = frames_dir / episode_history
    pkls = sorted(folder.glob("*.pkl"))
    if not pkls:
        return None

    positions = np.stack([_hm3d_camera_position(p) for p in pkls])
    idx = sorted(_fps_indices(positions, num_frames))
    images = [str(p.with_name(p.stem + "-rgb.png")) for p in (pkls[i] for i in idx)]
    if not all(Path(img).exists() for img in images):
        return None
    return _pad(images, num_frames)


# ---------------------------------------------------------------------------
# scannet-v0: pose from plain per-frame <frame>.txt (camera-to-world 4x4)
# ---------------------------------------------------------------------------

def _scannet_scene_id(episode_history: str) -> str:
    # "scannet-v0/002-scannet-scene0709_00" -> "scene0709_00"
    tail = episode_history.split("/", 1)[-1]
    return tail.split("-scannet-")[-1]


def sample_scannet_episode(scannet_dir: Path, episode_history: str, num_frames: int) -> list[str] | None:
    scene_id = _scannet_scene_id(episode_history)
    folder = scannet_dir / "posed_images" / scene_id
    pose_files = sorted(folder.glob("*.txt"))
    if not pose_files:
        return None

    positions, kept = [], []
    for p in pose_files:
        mat = np.loadtxt(p).reshape(4, 4)
        if not np.isfinite(mat).all():  # ScanNet convention: bad tracking -> all -inf
            continue
        positions.append(mat[:3, 3])
        kept.append(p)
    if not kept:
        return None

    idx = sorted(_fps_indices(np.stack(positions), num_frames))
    images = [str(kept[i].with_suffix(".jpg")) for i in idx]
    if not all(Path(img).exists() for img in images):
        return None
    return _pad(images, num_frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openeqa-dir", default="./source_data/openeqa", help="Dir with open-eqa-v0.json")
    parser.add_argument("--frames-dir", default=None, help="default: <openeqa-dir>/frames (hm3d-v0/* episodes)")
    parser.add_argument(
        "--scannet-dir", default="./source_data/scannet",
        help="Dir with posed_images/<scene_id>/<frame>.{jpg,txt} "
             "(dataset/download_scannet_posed.sh output). "
             "scannet-v0/* rows are skipped if not found.",
    )
    parser.add_argument("--out", default="./data/openeqa_manifest.jsonl")
    parser.add_argument("--num-frames", type=int, default=8)
    args = parser.parse_args()

    openeqa_dir = Path(args.openeqa_dir)
    frames_dir = Path(args.frames_dir or openeqa_dir / "frames")
    scannet_dir = Path(args.scannet_dir)
    src = openeqa_dir / "open-eqa-v0.json"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(src, encoding="utf-8") as f:
        rows = json.load(f)

    hm3d_reader_ok: bool | None = None  # lazy: only probe if a hm3d-v0 row is actually hit
    scannet_dir_checked = False
    scannet_dir_ok = False

    written, skipped = 0, 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rows:
            episode_history = row["episode_history"]

            if episode_history.startswith("hm3d-v0/"):
                if hm3d_reader_ok is None:
                    hm3d_reader_ok = hm3d_pose_reader_available(frames_dir, episode_history)
                frame_paths = (
                    sample_hm3d_episode(frames_dir, episode_history, args.num_frames)
                    if hm3d_reader_ok else None
                )
            elif episode_history.startswith("scannet-v0/"):
                if not scannet_dir_checked:
                    scannet_dir_checked = True
                    scannet_dir_ok = (scannet_dir / "posed_images").exists()
                    if not scannet_dir_ok:
                        print(
                            f"NOTE: no {scannet_dir}/posed_images/ -- all scannet-v0/* rows will be skipped"
                        )
                frame_paths = (
                    sample_scannet_episode(scannet_dir, episode_history, args.num_frames)
                    if scannet_dir_ok else None
                )
            else:
                frame_paths = None

            if frame_paths is None:
                skipped += 1
                continue

            record = {
                "id": row["question_id"],
                "images": frame_paths,
                "query": row["question"],
                "target": row["answer"],
                "question_type": row.get("category"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written: {written}, Skipped (frames/pose unavailable): {skipped}")
    print(f"Manifest saved to: {out_path}")


if __name__ == "__main__":
    main()
