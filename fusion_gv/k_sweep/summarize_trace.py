"""Summarize a torch.profiler Chrome-trace JSON (from tensorboard_trace_handler)
into kernel-time / occupancy / memory numbers -- pure stdlib (json, gzip),
no torch_tb_profiler or TensorBoard needed to read it.

The exact `args` key names used for occupancy below (e.g.
"est. achieved occupancy %") are inferred from what torch_tb_profiler's UI
displays ("Mean Est. Achieved Occupancy (%)") -- NOT independently confirmed
against the raw JSON schema. Run --inspect on one real trace file first and
check the printed sample keys match what this script looks for; if they
don't, fix _OCCUPANCY_KEYS below before trusting the sweep's numbers.

Usage:
    python3 summarize_trace.py <trace.pt.trace.json[.gz]>              # summary
    python3 summarize_trace.py <trace.pt.trace.json[.gz]> --inspect    # verify field names first
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

# Candidate keys for occupancy in a GPU kernel event's "args" dict -- tried
# in order, first match wins. Confirm with --inspect before trusting results.
_OCCUPANCY_KEYS = (
    "est. achieved occupancy %",
    "Est. Achieved Occupancy %",
    "achieved occupancy %",
    "occupancy",
)
_MEM_TOTAL_KEYS = ("Total Allocated", "Total Reserved", "Bytes")

# torch.profiler.record_function(...) labels added in fusion_gv/gvjepa.py's
# _pool_visual/_run_predictor -- summed regardless of `cat`, so we can see
# how much of the step is x_encoder (frozen VGGT/JEPA, K-independent) vs
# visual_resampler/predictor_forward (the K-dependent parts), isolated from
# each other. This is the number that actually answers "what does K cost",
# not the aggregate kernel totals (those are dominated by x_encoder).
_LABELED_REGIONS = ("x_encoder", "visual_resampler", "predictor_forward")


def _load_events(trace_file: Path) -> list[dict]:
    opener = gzip.open if trace_file.suffix == ".gz" else open
    with opener(trace_file, "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", data) if isinstance(data, dict) else data


def inspect(trace_file: Path, n: int = 5) -> None:
    """Print category set + a few sample GPU-kernel-looking events' arg keys,
    so you can confirm _OCCUPANCY_KEYS actually matches this trace's schema."""
    events = _load_events(trace_file)
    cats = sorted({e.get("cat", "") for e in events})
    print(f"{len(events)} events, categories: {cats}")

    shown = 0
    for e in events:
        cat = (e.get("cat") or "").lower()
        if "kernel" in cat and e.get("args"):
            print(f"\n--- sample kernel event (cat={e.get('cat')!r}) ---")
            print("name:", e.get("name"))
            print("dur:", e.get("dur"))
            print("args keys:", list(e["args"].keys()))
            shown += 1
            if shown >= n:
                break
    if shown == 0:
        print("No kernel-category events with args found -- check the 'kernel' "
              "substring match above, or dump a few raw events manually.")


def summarize(trace_file: Path, top_n: int = 10) -> dict:
    events = _load_events(trace_file)

    kernels: dict[str, dict] = {}
    peak_mem_bytes = 0.0
    labeled_us = {name: 0.0 for name in _LABELED_REGIONS}
    labeled_n = {name: 0 for name in _LABELED_REGIONS}

    for e in events:
        name0 = e.get("name")
        if name0 in _LABELED_REGIONS:
            labeled_us[name0] += e.get("dur", 0) or 0
            labeled_n[name0] += 1

        cat = (e.get("cat") or "").lower()
        if "kernel" in cat:
            name = e.get("name", "?")
            dur = e.get("dur", 0) or 0
            args = e.get("args") or {}
            occ = next((args[k] for k in _OCCUPANCY_KEYS if k in args), None)

            k = kernels.setdefault(name, {"dur": 0.0, "count": 0, "occ_sum": 0.0, "occ_n": 0})
            k["dur"] += dur
            k["count"] += 1
            if occ is not None:
                k["occ_sum"] += float(occ)
                k["occ_n"] += 1
        elif "memory" in cat:
            args = e.get("args") or {}
            total = next((args[k] for k in _MEM_TOTAL_KEYS if k in args), None)
            if total:
                peak_mem_bytes = max(peak_mem_bytes, float(total))

    total_kernel_time = sum(k["dur"] for k in kernels.values())
    ranked = sorted(kernels.items(), key=lambda kv: -kv[1]["dur"])[:top_n]
    top_name, top = ranked[0] if ranked else (None, None)
    top_occ = (top["occ_sum"] / top["occ_n"]) if top and top["occ_n"] else None

    weighted_sum = sum(k["occ_sum"] / k["occ_n"] * k["dur"] for k in kernels.values() if k["occ_n"])
    weight_total = sum(k["dur"] for k in kernels.values() if k["occ_n"])
    avg_occupancy_pct = (weighted_sum / weight_total) if weight_total else None

    return {
        "trace_file": str(trace_file),
        "trace_file_size_mb": round(trace_file.stat().st_size / 1e6, 1),
        "num_distinct_kernels": len(kernels),
        "total_kernel_time_us": total_kernel_time,
        "top_kernel_name": top_name,
        "top_kernel_time_us": top["dur"] if top else None,
        "top_kernel_occupancy_pct": top_occ,
        "avg_occupancy_pct_weighted": avg_occupancy_pct,
        # From torch.cuda.max_memory_allocated(), written by gvjepa_trainer.py
        # to <output_dir>/peak_memory.json -- more reliable than this field,
        # which is a best-effort raw-JSON guess (see _MEM_TOTAL_KEYS above).
        # run_sweep.py prefers peak_memory.json and only falls back to this.
        "peak_memory_gb_from_trace_guess": (peak_mem_bytes / 1e9) if peak_mem_bytes else None,
        "x_encoder_time_us": labeled_us["x_encoder"] or None,
        "visual_resampler_time_us": labeled_us["visual_resampler"] or None,
        "predictor_forward_time_us": labeled_us["predictor_forward"] or None,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    if "--inspect" in sys.argv:
        inspect(path)
    else:
        print(json.dumps(summarize(path), indent=2))
