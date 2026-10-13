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

VGGT-only, 2 GPUs, find the largest K that fits (see run_sweep_vggt_2gpu.sbatch):
    python3 fusion_gv/k_sweep/run_sweep.py \
        --config fusion_gv/configs/overfit_from_vljepa_ckpt_vggt.yaml \
        --out-root outputs/k_sweep_vggt_2gpu --num-gpus 2 \
        --overfit-samples 64 --k-values 32,64,96,128,192,256 --stop-on-fail
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


def run_one(k: int, config: str, out_root: Path, num_gpus: int, overfit_samples: int) -> dict:
    out_dir = out_root / f"k{k}"
    trace_dir = out_dir / "profiler_trace"
    shutil.rmtree(trace_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    launch = ["accelerate", "launch"]
    if num_gpus > 1:
        launch.append("--multi_gpu")
    cmd = launch + [
        "--num_processes", str(num_gpus),
        "fusion_gv/train_gvjepa.py",
        "--config", config,
        "--output-dir", str(out_dir),
        "--visual-pool-k", str(k),
        "--overfit",
        # _build_overfit_loader freezes this many samples as ONE batch per
        # process, so this is the per-GPU batch size, not the yaml's.
        "--overfit-samples", str(overfit_samples),
        "--overfit-steps", str(PROFILE_STEPS + 20),
        "--profile",
        "--profile-steps", str(PROFILE_STEPS),
    ]
    # stderr goes to a file so an OOM can be told apart from any other crash.
    stderr_log = out_dir / "stderr.log"
    with open(stderr_log, "w") as err:
        result = subprocess.run(cmd, stderr=err)
    # The trainer always writes final.pt (full model + optimizer state); a
    # sweep only needs the memory/timing numbers, so don't keep one per K.
    (out_dir / "final.pt").unlink(missing_ok=True)
    if result.returncode != 0:
        err_text = stderr_log.read_text(errors="replace")
        sys.stderr.write(err_text[-3000:])
        status = "oom" if "out of memory" in err_text.lower() else "failed"
        return {"k": k, "status": status, "returncode": result.returncode}

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
    parser.add_argument("--config", type=str, default=BASE_CONFIG,
                         help=f"Base YAML config (default: {BASE_CONFIG})")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT,
                         help=f"Per-K output dirs + k_sweep_results.json go here (default: {OUT_ROOT})")
    parser.add_argument("--num-gpus", type=int, default=1,
                         help="accelerate processes; peak memory is reported for the main process's GPU (default: 1)")
    parser.add_argument("--overfit-samples", type=int, default=32,
                         help="Per-GPU batch size of the frozen overfit batch (default: 32)")
    parser.add_argument("--stop-on-fail", action="store_true",
                         help="Stop at the first failed K instead of continuing through the rest")
    args = parser.parse_args()
    k_values = [int(x) for x in args.k_values.split(",")] if args.k_values else K_VALUES

    args.out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for k in k_values:
        print(f"\n=== K={k} ===", flush=True)
        r = run_one(k, args.config, args.out_root, args.num_gpus, args.overfit_samples)
        print(json.dumps(r, indent=2), flush=True)
        results.append(r)
        with open(args.out_root / "k_sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)
        if args.stop_on_fail and r["status"] != "ok":
            break

    _print_table(results)


if __name__ == "__main__":
    main()
