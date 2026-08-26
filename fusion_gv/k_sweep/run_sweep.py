"""K-sweep driver for VisualResampler (fusion_gv/gvjepa.py::VisualResampler).

Runs the overfit fusiongv --profile training at K in K_VALUES, each in its
own output dir under outputs/k_sweep/k<K>/, then summarizes each run's
profiler trace (summarize_trace.py) into one growth-curve table.

A failed run (OOM, crash) is recorded as such and does NOT stop the sweep --
the whole point is to find where it breaks.

Usage (inside the container, single GPU):
    python3 fusion_gv/k_sweep/run_sweep.py
    python3 fusion_gv/k_sweep/run_sweep.py --k-values 1,2,4,8
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from summarize_trace import summarize  # noqa: E402

K_VALUES = [1, 2, 4, 8, 12, 16, 24, 32]
BASE_CONFIG = "fusion_gv/configs/profile_gvjepa_fusiongv.yaml"
OUT_ROOT = Path("outputs/k_sweep")
PROFILE_STEPS = 10


def _latest_trace_file(trace_dir: Path) -> Path | None:
    files = sorted(trace_dir.glob("*.pt.trace.json*"))
    return files[-1] if files else None


def run_one(k: int) -> dict:
    out_dir = OUT_ROOT / f"k{k}"
    trace_dir = out_dir / "profiler_trace"
    shutil.rmtree(trace_dir, ignore_errors=True)

    cmd = [
        "accelerate", "launch", "--num_processes", "1",
        "fusion_gv/train_gvjepa.py",
        "--config", BASE_CONFIG,
        "--output-dir", str(out_dir),
        "--visual-pool-k", str(k),
        "--overfit",
        "--overfit-samples", "32",
        "--overfit-steps", str(PROFILE_STEPS + 20),
        "--profile",
        "--profile-steps", str(PROFILE_STEPS),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return {"k": k, "status": "failed", "returncode": result.returncode}

    trace_file = _latest_trace_file(trace_dir)
    if trace_file is None:
        return {"k": k, "status": "no_trace_file"}

    summary = summarize(trace_file)
    summary.update({"k": k, "status": "ok"})
    return summary


def _print_table(results: list[dict]) -> None:
    print(f"\n{'K':>4} {'status':<14} {'top_occ%':>9} {'avg_occ%':>9} {'kernel_us':>12} {'peak_mem_gb':>12}")
    for r in results:
        def fmt(v):
            return f"{v:.1f}" if isinstance(v, (int, float)) else "-"
        print(
            f"{r.get('k'):>4} {r.get('status', ''):<14} "
            f"{fmt(r.get('top_kernel_occupancy_pct')):>9} "
            f"{fmt(r.get('avg_occupancy_pct_weighted')):>9} "
            f"{fmt(r.get('total_kernel_time_us')):>12} "
            f"{fmt(r.get('peak_memory_gb')):>12}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="K-sweep for VisualResampler")
    parser.add_argument("--k-values", type=str, default=None,
                         help="Comma-separated K values, e.g. '1,2,4,8' (default: 1,2,4,8,12,16,24,32)")
    args = parser.parse_args()
    k_values = [int(x) for x in args.k_values.split(",")] if args.k_values else K_VALUES

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for k in k_values:
        print(f"\n=== K={k} ===", flush=True)
        r = run_one(k)
        print(json.dumps(r, indent=2), flush=True)
        results.append(r)
        with open(OUT_ROOT / "k_sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

    _print_table(results)


if __name__ == "__main__":
    main()
