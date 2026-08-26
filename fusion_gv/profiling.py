"""Opt-in torch.profiler helper.

Disabled by default: `maybe_profile(enabled=False, ...)` returns
`contextlib.nullcontext()` — no profiler object is ever constructed, so the
train/infer loops it wraps run exactly as before with zero added overhead.
Only when a caller explicitly passes `--profile` does this build a real
`torch.profiler.profile` context.

Usage:
    with maybe_profile(args.profile, trace_dir, active_steps=args.profile_steps) as prof:
        for step, batch in enumerate(loader):
            ...
            prof_step(prof)   # no-op when prof is None
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch


def maybe_profile(
    enabled: bool,
    trace_dir: str | Path,
    active_steps: int = 10,
    wait: int = 1,
    warmup: int = 1,
):
    """Returns a `torch.profiler.profile` context manager when `enabled`,
    else `contextlib.nullcontext()`. Callers use the same `with ... as prof:`
    line unconditionally; `prof` is `None` in the disabled case.

    Trace covers `wait` + `warmup` + `active_steps` iterations of `prof.step()`
    then stops recording (the `with` block may keep running past that, at
    ordinary — non-profiled — cost). Exported as a TensorBoard trace under
    `trace_dir`; open with `tensorboard --logdir <trace_dir>`.
    """
    if not enabled:
        return contextlib.nullcontext()
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active_steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def prof_step(prof) -> None:
    """`prof.step()`, or no-op when `prof` is None (the disabled/nullcontext case)."""
    if prof is not None:
        prof.step()
