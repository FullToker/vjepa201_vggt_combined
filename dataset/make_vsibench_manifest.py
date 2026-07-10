#!/usr/bin/env python3
"""Build a VL-JEPA inference manifest from a downloaded VSI-Bench test.jsonl.

Multiple-choice rows only (--mode mc, default): keeps rows with an `options`
list, maps the ground-truth letter (A/B/C/D) to the full option string, and
emits a `candidates` field so the eval side can score via candidate cosine
matching (no y-decoder needed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi-dir", default="./source_data/vsibench", help="Dir with test.jsonl and video subdirs")
    parser.add_argument("--out", default="./data/vsibench_manifest.jsonl")
    parser.add_argument(
        "--mode", choices=["mc", "open", "all"], default="mc",
        help="mc=multiple-choice only, open=open-ended only, all=both",
    )
    args = parser.parse_args()

    vsi_dir = Path(args.vsi_dir)
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

            video_path = vsi_dir / row["dataset"] / f"{row['scene_name']}.mp4"
            if not video_path.exists():
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
                "video": str(video_path),
                "query": row["question"],
                "target": target,
                "question_type": row["question_type"],
            }
            if candidates:
                record["candidates"] = candidates

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written: {written}, Skipped (missing video): {skipped}")
    print(f"Manifest saved to: {out_path}")


if __name__ == "__main__":
    main()
