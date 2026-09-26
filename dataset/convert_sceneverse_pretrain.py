#!/usr/bin/env python3
"""Builds the scene-level pretraining manifests (multi-view set -> scene description).

Sources (all already on disk, see dataset/download_*.sh):
  ScanNet  captions: source_data/sceneverse/scannet_scene_cap.json   {scan_id: {"captions": [...]}}
  3RScan   captions: source_data/leo_annotations/annotations/alignment/scene_caption/3rscan_scenecap_{train,val}.json
                     {uuid: [{"response": caption, ...}, ...]}
  3RScan   metadata: source_data/3rscan/3RScan.json (reference scan + rescans per environment)
  frames:            source_data/scannet/posed_images/<scan_id>/*.jpg
                     source_data/3rscan/<uuid>/sequence/frame-*.color.jpg

Each output row is one training sample:
  {"images": [8 abs paths], "query": "", "target": "<scene caption>",
   "group_id": "scannet/scene0442", "scan_id": "scene0442_00"}

  * rows per scan: max(--train-sets-per-scan, number of captions the scan has), each row a
    different frame set with its own caption, so every caption is used at least once (3RScan
    has ~15 per scan, ScanNet 3); with fewer captions than rows they are cycled.
    Validation uses a fixed --val-sets-per-scan.
  * images: the scan's frame list is cut into --num-frames equal segments and one frame is
    drawn at random from each, so a set spans the whole scan instead of clustering on
    neighbouring near-duplicate frames. Kept in temporal order; the model shuffles them.
  * group_id: the physical space, not the scan. ScanNet sceneXXXX_YY -> sceneXXXX (YY is a
    rescan index); 3RScan -> the environment's reference-scan id from 3RScan.json. Rescans
    of one room share it, which the loss uses to avoid treating them as negatives.
  * split: by group_id (hash), so rescans of one room never land on both sides.

Usage:
  python3 dataset/convert_sceneverse_pretrain.py --root /glob/g01-cache/pf/Yushuo/vjepa201
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def scannet_group(scan_id: str) -> str:
    return "scannet/" + scan_id.split("_")[0]


def load_rscan_env_map(meta_path: Path) -> dict:
    env_of = {}
    for entry in json.load(open(meta_path)):
        ref = entry["reference"]
        env_of[ref] = ref
        for scan in entry.get("scans", []):
            env_of[scan["reference"]] = ref
    return env_of


def is_val(group: str, fraction: float, seed: int) -> bool:
    h = int(hashlib.md5(f"{seed}:{group}".encode()).hexdigest(), 16)
    return (h % 10000) < fraction * 10000


def stratified_frames(frames: list, n: int, rng: random.Random) -> list:
    """One random frame from each of n equal segments of the frame list (needs len(frames) >= n)."""
    bounds = [i * len(frames) // n for i in range(n + 1)]
    return [frames[rng.randrange(bounds[i], bounds[i + 1])] for i in range(n)]


def scannet_frames(posed_root: Path, scan_id: str) -> list:
    return sorted(str(p) for p in (posed_root / scan_id).glob("*.jpg") if "-annotated" not in p.stem)


def rscan_frames(rscan_root: Path, uuid: str) -> list:
    return sorted(str(p) for p in (rscan_root / uuid / "sequence").glob("frame-*.color.jpg"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="project root holding source_data/ and data/")
    ap.add_argument("--output-dir", type=Path, default=None, help="default: <root>/data")
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--train-sets-per-scan", type=int, default=8,
                    help="minimum frame sets per scan (train); scans with more captions than this get one row per caption")
    ap.add_argument("--val-sets-per-scan", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.05, help="fraction of spaces (not scans) held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-output", default="sceneverse_scene_pretrain_train.jsonl")
    ap.add_argument("--val-output", default="sceneverse_scene_pretrain_val.jsonl")
    args = ap.parse_args()

    root = args.root.resolve()
    src = root / "source_data"
    out_dir = args.output_dir or root / "data"
    scannet_cap_path = src / "sceneverse/scannet_scene_cap.json"
    leo_dir = src / "leo_annotations/annotations/alignment/scene_caption"
    posed_root = src / "scannet/posed_images"
    rscan_root = src / "3rscan"
    rscan_meta = rscan_root / "3RScan.json"

    for p in (scannet_cap_path, leo_dir, posed_root, rscan_root):
        if not p.exists():
            sys.exit(f"missing: {p}")
    if not rscan_meta.exists():
        sys.exit(f"missing: {rscan_meta} -- 3RScan environment grouping needs it "
                 "(http://campar.in.tum.de/public_datasets/3RScan/3RScan.json; notebooks/rescan_check.ipynb fetches it)")

    scannet_caps = {sid: v["captions"] for sid, v in json.load(open(scannet_cap_path)).items()}
    rscan_caps = {}
    for split in ("train", "val"):   # leo's own split is by scan; we re-split by space below
        for uuid, entries in json.load(open(leo_dir / f"3rscan_scenecap_{split}.json")).items():
            rscan_caps[uuid] = [e["response"] for e in entries]
    env_of = load_rscan_env_map(rscan_meta)

    sources = [
        ("scannet", scannet_caps, lambda s: scannet_group(s), lambda s: scannet_frames(posed_root, s)),
        ("3rscan", rscan_caps, lambda s: "3rscan/" + env_of.get(s, s), lambda s: rscan_frames(rscan_root, s)),
    ]

    rows = {"train": [], "val": []}
    scans_used = defaultdict(int)
    groups_seen = {"train": set(), "val": set()}
    skipped = defaultdict(int)

    for name, caps, group_of, frames_of in sources:
        for scan_id in sorted(caps):
            group = group_of(scan_id)
            frames = frames_of(scan_id)
            if len(frames) < args.num_frames:
                skipped[f"{name}: fewer than {args.num_frames} frames"] += 1
                continue
            if not caps[scan_id]:
                skipped[f"{name}: no captions"] += 1
                continue

            split = "val" if is_val(group, args.val_fraction, args.seed) else "train"
            n_sets = args.val_sets_per_scan if split == "val" else max(args.train_sets_per_scan, len(caps[scan_id]))
            rng = random.Random(f"{args.seed}:{scan_id}")   # per-scan, so output is independent of scan order
            captions = list(caps[scan_id])
            rng.shuffle(captions)                            # then cycle, so every caption gets used evenly
            for k in range(n_sets):
                rows[split].append({
                    "images": stratified_frames(frames, args.num_frames, rng),
                    "query": "",
                    "target": captions[k % len(captions)],
                    "group_id": group,
                    "scan_id": scan_id,
                })
            scans_used[f"{name}/{split}"] += 1
            groups_seen[split].add(group)

    assert not (groups_seen["train"] & groups_seen["val"]), "a space landed in both splits"

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, fname in (("train", args.train_output), ("val", args.val_output)):
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            for row in rows[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows[split])} rows, {len(groups_seen[split])} spaces -> {out_dir / fname}")
    print("scans used:", dict(scans_used))
    if skipped:
        print("skipped:", dict(skipped))


if __name__ == "__main__":
    main()
