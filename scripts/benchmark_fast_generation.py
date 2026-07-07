"""Benchmark LTISystem.generate_dataset: fast parallel scan vs the generic loop.

Times the fast path (``LTISystem.generate_dataset``) against the generic per-step
rollout (``SystemGroup.generate_dataset``) on an estimation-only (LQE) system for
a few sequence lengths, on CPU and (if available) CUDA. Both produce the same
fields/shape; only the sampling implementation differs.

Run: PYTHONPATH=. python scripts/benchmark_fast_generation.py
"""
import time

import torch

from kf_rnn.infrastructure.config.schema import EnvironmentShape, ProblemShape, SystemConfig
from kf_rnn.system.base import SystemGroup
from kf_rnn.system.linear_time_invariant import ContinuousDistribution


def _build(S_D, O_D):
    ps = ProblemShape(environment=EnvironmentShape(observation=O_D), controller={})
    cfg = SystemConfig(S_D=S_D, problem_shape=ps)
    dist = ContinuousDistribution("gaussian", "gaussian", 0.1, 1.0, 1.0)
    return dist.sample(cfg, ())


def _time(fn, device, repeats, warmup=True):
    # optional warmup, then median of `repeats`
    if warmup:
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def benchmark(device):
    torch.set_default_device(device)
    torch.manual_seed(0)
    S_D, O_D, N = 6, 2, 16
    print(f"\n=== device={device}  S_D={S_D} O_D={O_D} N={N} ===")
    print(f"{'L':>8} {'loop (s)':>12} {'fast (s)':>12} {'speedup':>10}")
    for L in (200, 2000, 10000):
        lsg = _build(S_D, O_D)
        # The loop is O(L) Python steps and dominates the wall time, so time it
        # once with no warmup; the fast path is cheap, so warm it up and median it.
        t_loop = _time(lambda: SystemGroup.generate_dataset(lsg, N, L), device, 1, warmup=False)
        t_fast = _time(lambda: lsg.generate_dataset(N, L), device, 5)
        print(f"{L:>8} {t_loop:>12.4f} {t_fast:>12.4f} {t_loop / t_fast:>9.1f}x", flush=True)


def main():
    benchmark("cpu")
    if torch.cuda.is_available():
        benchmark("cuda")
    else:
        print("\n(CUDA not available; skipped)")


if __name__ == "__main__":
    main()
