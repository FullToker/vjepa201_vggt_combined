#!/usr/bin/env python3
"""Build a fusion_gv inference manifest from dataset/extract_mmsibench_images.py output.

Parses the A/B/C/D options embedded in each row's `question` text into a
`candidates` list (for cosine-candidate scoring, same as VSI-Bench), and maps
the ground-truth `answer` letter to the matching candidate as `target`.

MMSI-Bench rows have 2-10 images each (no fixed frame count like VSI-Bench's
8), so rows are sorted into contiguous same-image-count buckets instead.
Downstream batch construction should slice each bucket independently: the
last partial chunk of a bucket just runs at reduced batch size -- never pad
or mix rows across bucket boundaries (collate requires uniform S per batch).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

# Options are inline, comma-joined on one line: "Options: A: back left, B:
# front left, C: front right, D: back right" -- not one-per-line. Find each
# "<letter>:" marker and slice the text up to the next marker (or EOL).
# Lookbehind blocks mid-word letters (e.g. "3:00") from matching.
_OPTION_MARKER_RE = re.compile(r"(?<![A-Za-z])([A-D]):\s*")


def _parse_options(question: str) -> Optional[Dict[str, str]]:
    matches = list(_OPTION_MARKER_RE.finditer(question))
    if len(matches) < 2:
        return None

    options: Dict[str, str] = {}
    for i, m in enumerate(matches):
        letter = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        text = question[start:end].strip().rstrip(",").strip()
        if letter not in options and text:
            options[letter] = text
    return options if len(options) >= 2 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", default="./data/mmsibench_raw.jsonl")
    parser.add_argument("--out", default="./data/mmsibench_manifest.jsonl")
    args = parser.parse_args()

    src = Path(args.raw_jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    with open(src, encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            options = _parse_options(row["question"])
            if options is None:
                skipped += 1
                continue

            letter = str(row["answer"]).strip().upper()
            if letter not in options:
                skipped += 1
                continue

            candidates = [f"{l}. {options[l]}" for l in sorted(options)]
            target = f"{letter}. {options[letter]}"

            rows.append(
                {
                    "id": row["id"],
                    "images": row["images"],
                    "num_images": row["num_images"],
                    "query": row["question"],
                    "target": target,
                    "candidates": candidates,
                    "question_type": row.get("question_type"),
                    "difficulty": row.get("difficulty"),
                }
            )

    # Bucket by image count so sequential batch slicing stays same-S per batch.
    rows.sort(key=lambda r: (r["num_images"], r["id"]))

    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    bucket_counts = Counter(r["num_images"] for r in rows)
    print(f"Written: {len(rows)}, Skipped (unparsed options / bad answer letter): {skipped}")
    print(f"Manifest saved to: {out_path}")
    print("Image-count buckets (num_images -> row count):")
    for n in sorted(bucket_counts):
        print(f"  {n}: {bucket_counts[n]}")


if __name__ == "__main__":
    main()
