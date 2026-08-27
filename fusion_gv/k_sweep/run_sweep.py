"""K-sweep driver for VisualResampler (fusion_gv/gvjepa.py::VisualResampler).

Runs the overfit fusiongv --profile training at K in K_VALUES (default up to
1369 = full patch grid P, i.e. no pooling at all), each in its own output
dir under outputs/k_sweep/k<K>/, then summarizes each run's profiler trace
(summarize_trace.py) into one growth-curve table.

Records, per K:
  - peak_memory_gb: real torch.cuda.max_memory_allocated(), written by
    gvjepa_trainer.py to <output_dir>/peak_memory.json
  - x_encoder / visual_resampler / predictor_forward time: isolated via
    torch.profiler.record_function labels in gvjepa.py, so K's actual cost
    is visible separately from the (K-independent) frozen VGGT/JEPA forward
    that otherwise dominates and hides it (see chat history).

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

K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1369]  # 1369 = P, full patch grid
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

    # Real peak memory, from torch.cuda.max_memory_allocated() written by
    # gvjepa_trainer.py -- preferred over summarize()'s raw-JSON guess.
    mem_file = out_dir / "peak_memory.json"
    if mem_file.exists():
        summary["peak_memory_gb"] = json.loads(mem_file.read_text())["peak_memory_gb"]
    else:
        summary["peak_memory_gb"] = summary.get("peak_memory_gb_from_trace_guess")

    summary.update({"k": k, "status": "ok"})
    return summary


def _print_table(results: list[dict]) -> None:
    print(
        f"\n{'K':>5} {'status':<14} {'peak_mem_gb':>11} {'avg_occ%':>9} "
        f"{'x_encoder_us':>13} {'resampler_us':>13} {'predictor_us':>13}"
    )
    for r in results:
        def fmt(v):
            return f"{v:.1f}" if isinstance(v, (int, float)) else "-"
        print(
            f"{r.get('k'):>5} {r.get('status', ''):<14} "
            f"{fmt(r.get('peak_memory_gb')):>11} "
            f"{fmt(r.get('avg_occupancy_pct_weighted')):>9} "
            f"{fmt(r.get('x_encoder_time_us')):>13} "
            f"{fmt(r.get('visual_resampler_time_us')):>13} "
            f"{fmt(r.get('predictor_forward_time_us')):>13}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="K-sweep for VisualResampler")
    parser.add_argument("--k-values", type=str, default=None,
                         help="Comma-separated K values, e.g. '1,2,4,8' (default: 1,2,4,8,16,32,64,128,256,512,768,1024,1369)")
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
