"""Side-by-side diff of two or more infer_vsibench.py metrics files.

infer_vsibench.py writes <output_dir>/vsibench_metrics.json per run. This
collates several of them into one table: overall candidate-match accuracy,
per-question-type accuracy, and the embedding-space metrics -- one column per
run, plus a delta column vs a chosen baseline.

Pure stdlib. Missing / unreadable files become an all-"-" column instead of
crashing, so a partially-finished sweep still prints.

Usage:
    python3 fusion_gv/k_sweep/compare_vsibench.py \
        NAME=path/to/vsibench_metrics.json  [NAME2=path ...] \
        [--baseline NAME] [--out report.md]

Example:
    python3 fusion_gv/k_sweep/compare_vsibench.py \
        K1-32f=outputs/inference/sft_from_vljepa_ckpt_fusiongv/vsibench_32f/vsibench_metrics.json \
        K128-32f=outputs/inference/sft_from_vljepa_ckpt_fusiongv_k128/vsibench_32f/vsibench_metrics.json \
        --baseline K1-32f
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EMB_KEYS = (
    "held_out_infonce_loss",
    "alignment",
    "uniformity_pred",
    "uniformity_target",
    "auc",
    "margin_mean",
)


def _load(path: str) -> dict | None:
    p = Path(path)
    if not p.is_file():
        print(f"WARN: missing {p}", file=sys.stderr)
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARN: unreadable {p}: {e}", file=sys.stderr)
        return None


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _delta(v, base) -> str:
    if v is None or base is None:
        return "-"
    d = v - base
    return f"{d:+.4f}"


def _rows(runs: list[tuple[str, dict | None]], baseline: str):
    """Yield (label, {name: value_str}, is_section_header)."""
    names = [n for n, _ in runs]
    data = {n: (d or {}) for n, d in runs}
    base = data.get(baseline, {})

    # ── headline ────────────────────────────────────────────────────────────
    yield "OVERALL", None, True
    for key, nd, label in [
        ("accuracy", 4, "accuracy"),
        ("evaluated", 0, "evaluated (n)"),
        ("correct", 0, "correct"),
        ("total", 0, "total rows"),
        ("mean_sim_score", 4, "mean_sim_score"),
    ]:
        cells = {n: _fmt(data[n].get(key), nd) for n in names}
        if key == "accuracy":
            cells = {
                n: f"{_fmt(data[n].get(key))}  ({_delta(data[n].get(key), base.get(key))})"
                for n in names
            }
        yield label, cells, False

    # step / manifest provenance
    yield "step", {n: _fmt(data[n].get("step")) for n in names}, False
    yield "manifest", {n: Path(str(data[n].get("manifest", "-"))).name for n in names}, False

    # ── per question type ───────────────────────────────────────────────────
    qtypes = set()
    for n in names:
        qtypes.update((data[n].get("per_question_type") or {}).keys())
    if qtypes:
        yield "PER QUESTION TYPE (accuracy)", None, True
        for qt in sorted(qtypes):
            cells = {}
            for n in names:
                pqt = (data[n].get("per_question_type") or {}).get(qt) or {}
                acc = pqt.get("accuracy")
                bpqt = (base.get("per_question_type") or {}).get(qt) or {}
                cells[n] = f"{_fmt(acc)}  ({_delta(acc, bpqt.get('accuracy'))})"
            yield qt, cells, False

    # ── embedding metrics ───────────────────────────────────────────────────
    if any(data[n].get("embedding_metrics") for n in names):
        yield "EMBEDDING METRICS", None, True
        for key in _EMB_KEYS:
            cells = {}
            for n in names:
                em = data[n].get("embedding_metrics") or {}
                bem = base.get("embedding_metrics") or {}
                cells[n] = f"{_fmt(em.get(key))}  ({_delta(em.get(key), bem.get(key))})"
            yield key, cells, False
        yield "num_samples", {
            n: _fmt((data[n].get("embedding_metrics") or {}).get("num_samples")) for n in names
        }, False


def _render(runs, baseline: str) -> str:
    names = [n for n, _ in runs]
    all_rows = list(_rows(runs, baseline))
    w_label = max(
        [len("PER QUESTION TYPE (accuracy)")]
        + [len(n) for n in names]
        + [len(lbl) for lbl, cells, hdr in all_rows if not hdr]
    ) + 1
    w_col = max(
        [22]
        + [len(n) + 2 for n in names]
        + [len(str(cells[n])) + 2 for lbl, cells, hdr in all_rows if not hdr for n in names]
    )

    lines = []
    header = "metric".ljust(w_label) + "".join(n.rjust(w_col) for n in names)
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(f"(delta in parens vs baseline: {baseline})")
    lines.append("")
    for label, cells, is_header in all_rows:
        if is_header:
            lines.append("")
            lines.append(f"# {label}")
            continue
        lines.append(label.ljust(w_label) + "".join(str(cells[n]).rjust(w_col) for n in names))
    return "\n".join(lines)


def _render_md(runs, baseline: str) -> str:
    names = [n for n, _ in runs]
    lines = [f"# VSI-Bench K comparison (baseline: {baseline})", ""]
    lines.append("| metric | " + " | ".join(names) + " |")
    lines.append("|" + "---|" * (len(names) + 1))
    for label, cells, is_header in _rows(runs, baseline):
        if is_header:
            lines.append(f"| **{label}** | " + " | ".join([""] * len(names)) + " |")
            continue
        lines.append(f"| {label} | " + " | ".join(str(cells[n]) for n in names) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="NAME=PATH", help="named metrics json files")
    ap.add_argument("--baseline", default=None, help="run NAME to diff against (default: first)")
    ap.add_argument("--out", default=None, help="also write a markdown table here")
    args = ap.parse_args()

    runs: list[tuple[str, dict | None]] = []
    for spec in args.runs:
        if "=" not in spec:
            ap.error(f"expected NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        runs.append((name, _load(path)))

    baseline = args.baseline or runs[0][0]
    if baseline not in {n for n, _ in runs}:
        ap.error(f"--baseline {baseline!r} not among {[n for n, _ in runs]}")

    print(_render(runs, baseline))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_md(runs, baseline))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
