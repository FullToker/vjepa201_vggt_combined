#!/usr/bin/env python3
"""Build a fusion_gv inference manifest from a downloaded VSI-Bench test.jsonl.

Reads pre-extracted frame folders (dataset/extract_vsibench_frames.py output,
default <vsi-dir>/frames/<id>/frame_*.jpg) instead of raw video -- rows whose
frames haven't been extracted yet (or don't have exactly --num-frames) are
skipped, so this can be re-run anytime after a partial extraction pass.

Multiple-choice rows only (--mode mc, default): keeps rows with an `options`
list, maps the ground-truth letter (A/B/C/D) to the full option string, and
emits a `candidates` field so the eval side can score via candidate cosine
matching (no y-decoder needed).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _sanitize_id(value) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi-dir", default="./source_data/vsibench", help="Dir with test.jsonl")
    parser.add_argument("--frames-dir", default=None, help="default: <vsi-dir>/frames (extract_vsibench_frames.py output)")
    parser.add_argument("--out", default="./data/vsibench_manifest.jsonl")
    parser.add_argument("--num-frames", type=int, default=8, help="Must match extract_vsibench_frames.py --num-frames")
    parser.add_argument(
        "--mode", choices=["mc", "open", "all"], default="mc",
        help="mc=multiple-choice only, open=open-ended only, all=both",
    )
    args = parser.parse_args()

    vsi_dir = Path(args.vsi_dir)
    frames_dir = Path(args.frames_dir or vsi_dir / "frames")
    src = vsi_dir / "test.jsonl"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, 0
    with open(src, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            has_options = bool(row.get("options"))

            if args.mode == "mc" and not has_options:
                continue
            if args.mode == "open" and has_options:
                continue

            sample_dir = frames_dir / _sanitize_id(row["id"])
            frame_paths = sorted(
                sample_dir.glob("frame_*.jpg"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            if len(frame_paths) != args.num_frames:
                skipped += 1
                continue

            if has_options:
                letter = row["ground_truth"].strip().upper()
                target = next(
                    (opt for opt in row["options"] if opt.startswith(letter + ".")),
                    row["ground_truth"],
                )
                candidates = row["options"]
            else:
                target = row["ground_truth"]
                candidates = None

            record = {
                "id": row["id"],
                "images": [str(p) for p in frame_paths],
                "query": row["question"],
                "target": target,
                "question_type": row["question_type"],
            }
            if candidates:
                record["candidates"] = candidates

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written: {written}, Skipped (frames not extracted / wrong count): {skipped}")
    print(f"Manifest saved to: {out_path}")


if __name__ == "__main__":
    main()
