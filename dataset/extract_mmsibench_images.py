#!/usr/bin/env python3
"""Explode MMSI-Bench parquet (embedded images) into jpg files + a raw jsonl manifest.

Reads all *.parquet under --src-dir (as downloaded by download_mmsibench.sh),
writes each row's image bytes to <src-dir>/images/<id>/img_<idx>.jpg, and emits
one manifest line per row with the image bytes replaced by local file paths.

No multiple-choice / candidate parsing here -- MMSI-Bench bakes the options
into the `question` text with a single `answer` letter (A-D), and images-per-row
varies (2-10). Building the fixed-eval manifest (candidate extraction, frame
bucketing) is a separate downstream step.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image


def _sanitize_id(value) -> str:
    return str(value).replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", default="./source_data/mmsibench")
    parser.add_argument("--out-jsonl", default="./data/mmsibench_raw.jsonl")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    images_dir = src_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(glob.glob(str(src_dir / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise SystemExit(f"No parquet files found under {src_dir}")

    written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for pq in parquet_files:
            df = pd.read_parquet(pq)
            for row in df.to_dict("records"):
                sample_id = _sanitize_id(row["id"])
                sample_dir = images_dir / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)

                image_paths = []
                for idx, img in enumerate(row["images"]):
                    img_bytes = img["bytes"] if isinstance(img, dict) else img
                    out_img = sample_dir / f"img_{idx}.jpg"
                    if not out_img.exists():
                        Image.open(io.BytesIO(img_bytes)).convert("RGB").save(out_img, "JPEG")
                    image_paths.append(str(out_img))

                record = {
                    "id": row["id"],
                    "images": image_paths,
                    "num_images": len(image_paths),
                    "question": row["question"],
                    "answer": row["answer"],
                    "question_type": row.get("question_type"),
                    "difficulty": row.get("difficulty"),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"Written {written} rows -> {out_path}")
    print(f"Images saved under: {images_dir}")


if __name__ == "__main__":
    main()
