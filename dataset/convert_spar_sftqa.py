#!/usr/bin/env python3
"""Convert SPAR-7M-RGBD QA files into FusionGVJEPA SFT manifests.

The converter expects the extracted SPAR layout under ./source_data/spar:

  source_data/spar/{rxr,scannet,scannetpp,structured3d}/images/...
  source_data/spar/{rxr,scannet,scannetpp,structured3d}/qa_jsonl/...

Default split policy:
  - sentence + fill -> train manifest
  - select          -> eval manifest

Train rows:
  {"images": ["...", "...", "...", "..."], "query": "...", "target": "..."}

Eval/select rows:
  {"images": ["...", "...", "...", "..."], "query": "...", "candidates": [...], "target": "..."}

For select rows, target is converted from the option label (e.g. "A") to the
corresponding candidate text without the A/B/C/D prefix.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Sequence, TextIO

import numpy as np

TRAIN_KINDS = ("sentence", "fill")
EVAL_KINDS = ("select",)
OPTION_RE = re.compile(r"^\s*([A-Z])\s*[\.)]\s*(.+?)\s*$")
ANSWER_INSTRUCTION_RE = re.compile(
    r"^\s*Your answer (?:can|should|must).*?(?:options?\s+)?[A-Z](?:\s*,\s*[A-Z]|\s+or\s+[A-Z]|\s*)*[\.]?\s*$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SPAR-7M-RGBD LLaVA-style QA JSONL into FusionGVJEPA manifests."
    )
    parser.add_argument("--spar-root", default="./source_data/spar", help="Extracted SPAR root. Default: ./source_data/spar")
    parser.add_argument("--output-dir", default="./data", help="Output directory for generated JSONL manifests. Default: ./data")
    parser.add_argument("--train-output", default="spar_sftqa_train.jsonl", help="Train manifest filename under --output-dir.")
    parser.add_argument("--eval-output", default="spar_sftqa_eval.jsonl", help="Eval manifest filename under --output-dir.")
    parser.add_argument("--split", default="train", help="SPAR split to read. Default: train")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of RGB frames per output sample. Default: 8")
    parser.add_argument("--only", choices=("both", "train", "eval"), default="both", help="Which manifests to generate. Default: both")
    parser.add_argument("--max-train", type=int, default=None, help="Optional max number of train rows for smoke tests.")
    parser.add_argument("--max-eval", type=int, default=None, help="Optional max number of eval rows for smoke tests.")
    parser.add_argument("--no-verify-images", action="store_true", help="Skip checking whether resolved RGB files exist.")
    parser.add_argument("--include-metadata", action="store_true", help="Include source/id/type/qa_kind metadata fields in each output row.")
    return parser.parse_args()


def iter_qa_files(spar_root: Path, split: str, kinds: Sequence[str]) -> Iterable[tuple[str, Path]]:
    for kind in kinds:
        pattern = f"*/qa_jsonl/{split}/*/{kind}/*.jsonl"
        for path in sorted(spar_root.glob(pattern)):
            yield kind, path


def get_source_name(spar_root: Path, qa_path: Path) -> str:
    return qa_path.relative_to(spar_root).parts[0]


def first_conversation_value(conversations: object, role: str) -> str:
    if not isinstance(conversations, list):
        return ""
    for item in conversations:
        if isinstance(item, dict) and item.get("from") == role:
            return str(item.get("value", "")).strip()
    return ""


def parse_select_query_and_candidates(query: str, target: str) -> tuple[str, list[str], str] | None:
    """Split select prompt into clean query, candidate texts, and target text."""
    question_lines: list[str] = []
    options: dict[str, str] = {}

    for line in query.splitlines():
        match = OPTION_RE.match(line)
        if match:
            label, candidate = match.groups()
            options[label.upper()] = candidate.strip()
            continue
        if ANSWER_INSTRUCTION_RE.match(line):
            continue
        question_lines.append(line.rstrip())

    candidates = [text for _, text in sorted(options.items())]
    clean_query = "\n".join(question_lines).strip()
    target_clean = target.strip()

    if not clean_query or not candidates:
        return None

    if len(target_clean) == 1 and target_clean.upper() in options:
        target_text = options[target_clean.upper()]
    elif target_clean in candidates:
        target_text = target_clean
    else:
        return None

    return clean_query, candidates, target_text


def sample_frames(images: Sequence[str], num_frames: int) -> list[str]:
    """Return exactly num_frames image paths.

    If SPAR provides more images, sample uniformly across the sequence. If it
    provides fewer images, cycle them to keep fixed S for batching.
    """
    if num_frames <= 0:
        raise ValueError("--num-frames must be > 0")
    if not images:
        return []
    images = list(images)
    if len(images) >= num_frames:
        if num_frames == 1:
            return [images[len(images) // 2]]
        idxs = [round(i * (len(images) - 1) / (num_frames - 1)) for i in range(num_frames)]
        return [images[i] for i in idxs]
    return [images[i % len(images)] for i in range(num_frames)]


def normalize_image_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value if x]
    return []


def convert_one_file(
    spar_root: Path,
    qa_kind: str,
    qa_path: Path,
    out_f: TextIO,
    num_frames: int,
    verify_images: bool,
    include_metadata: bool,
    limit_remaining: int | None,
) -> tuple[int, int, int | None]:
    source = get_source_name(spar_root, qa_path)
    image_root = spar_root / source / "images"
    written = 0
    skipped = 0

    with qa_path.open(encoding="utf-8") as f:
        for line in f:
            if limit_remaining is not None and limit_remaining <= 0:
                break
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            query = first_conversation_value(raw.get("conversations"), "human")
            target = first_conversation_value(raw.get("conversations"), "gpt")
            rel_images = normalize_image_list(raw.get("image"))
            if not rel_images:
                rel_images = normalize_image_list(raw.get("sorted_image"))
            if not query or not target or not rel_images:
                skipped += 1
                continue

            selected = sample_frames(rel_images, num_frames)
            images = [str(image_root / rel) for rel in selected]
            if verify_images and not all(Path(path).exists() for path in images):
                skipped += 1
                continue

            if qa_kind == "select":
                parsed = parse_select_query_and_candidates(query, target)
                if parsed is None:
                    skipped += 1
                    continue
                query, candidates, target = parsed
                row = {"images": images, "query": query, "candidates": candidates, "target": target}
            else:
                row = {"images": images, "query": query, "target": target}

            if include_metadata:
                row.update({"source": source, "id": raw.get("id"), "type": raw.get("type"), "qa_kind": qa_kind})
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if limit_remaining is not None:
                limit_remaining -= 1

    return written, skipped, limit_remaining


def convert_group(
    spar_root: Path,
    split: str,
    kinds: Sequence[str],
    output_path: Path,
    num_frames: int,
    verify_images: bool,
    include_metadata: bool,
    max_rows: int | None,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    remaining = max_rows
    total_files = 0
    total_written = 0
    total_skipped = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for qa_kind, qa_path in iter_qa_files(spar_root, split, kinds):
            if remaining is not None and remaining <= 0:
                break
            total_files += 1
            written, skipped, remaining = convert_one_file(
                spar_root=spar_root,
                qa_kind=qa_kind,
                qa_path=qa_path,
                out_f=out_f,
                num_frames=num_frames,
                verify_images=verify_images,
                include_metadata=include_metadata,
                limit_remaining=remaining,
            )
            total_written += written
            total_skipped += skipped

    return {"output": str(output_path), "qa_kinds": list(kinds), "files": total_files, "written": total_written, "skipped": total_skipped}


def main() -> None:
    args = parse_args()
    spar_root = Path(args.spar_root)
    if not spar_root.exists():
        raise FileNotFoundError(f"SPAR root not found: {spar_root}")

    output_dir = Path(args.output_dir)
    verify_images = not args.no_verify_images
    result: dict[str, object] = {}

    if args.only in ("both", "train"):
        result["train"] = convert_group(
            spar_root=spar_root,
            split=args.split,
            kinds=TRAIN_KINDS,
            output_path=output_dir / args.train_output,
            num_frames=args.num_frames,
            verify_images=verify_images,
            include_metadata=args.include_metadata,
            max_rows=args.max_train,
        )
    if args.only in ("both", "eval"):
        result["eval"] = convert_group(
            spar_root=spar_root,
            split=args.split,
            kinds=EVAL_KINDS,
            output_path=output_dir / args.eval_output,
            num_frames=args.num_frames,
            verify_images=verify_images,
            include_metadata=args.include_metadata,
            max_rows=args.max_eval,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
