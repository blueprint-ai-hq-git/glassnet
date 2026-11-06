"""gpu_bench.py — CPU vs GPU on the REAL T2 workload (not a toy).

Uses the actual TinyLM3 model from the T2 runner: embedding -> concat window
(512-d) -> 24 sigmoid concepts -> 8001-way head, with the real sparsity terms
and the real data (tiny_shakespeare, 8-token windows, V=8001).

Each step does exactly what the trainer does: transfer the batch, forward,
backward, SGD step. So the measured time/step is directly comparable to the
seconds/epoch in runs/t2/step2/sweep.log.

NOTE ON HONESTY: if the CPU is busy with another job, the CPU number is
inflated and the GPU looks better than it is. This script prints the current
load average so a contended run is visible rather than silently flattering.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np

from glassnet.tensor import Tensor, set_backend, backend_name
from glassnet.optim import SGD
from glassnet import tiny_data
from run_t2_iter003 import TinyLM3

K, D, W, BATCH = 24, 64, 8, 128


def bench(backend, steps=20, seed=0):
    set_backend(backend)
    d = tiny_data.build(win=W)
    V = d["V"]
    X, y = d["Xtr"], d["ytr"]
    rng = np.random.default_rng(seed)
    net = TinyLM3(V, K, D, W, rng=rng)
    n_par = sum(int(p.data.size) for p in net.parameters())

    # warm-up (JIT / BLAS thread spin-up) — not timed
    opt = SGD(net.parameters(), lr=0.1)
    for s in range(3):
        idx = np.arange(s * BATCH, (s + 1) * BATCH) % len(X)
        opt.zero_grad()
        loss, _ = net.loss(X[idx], y[idx])
        loss.backward()
        opt.step()

    times = []
    for s in range(steps):
        a = (s + 3) * BATCH
        idx = np.arange(a, a + BATCH) % len(X)
        t0 = time.perf_counter()
        opt.zero_grad()
        loss, _ = net.loss(X[idx], y[idx])
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        "backend": backend_name(),
        "params": n_par,
        "median_ms": float(np.median(times) * 1000),
        "mean_ms": float(times.mean() * 1000),
        "steps": steps,
        "batch": BATCH,
    }


def main():
    with open("/proc/loadavg") as f:
        load = f.read().split()[:3]
    print("=" * 70)
    print("REAL T2 WORKLOAD BENCHMARK  (TinyLM3, V=8001, K=24, D=64, W=8)")
    print(f"load average now: {load[0]} {load[1]} {load[2]}  "
          f"(>1.0 per core means the CPU is contended)")
    print("=" * 70)

    cpu = bench("numpy")
    gpu = bench("cupy")

    print(f"\n  CPU ({cpu['backend']}): {cpu['median_ms']:.1f} ms/step (median), "
          f"{cpu['mean_ms']:.1f} mean, {cpu['params']} params")
    print(f"  GPU ({gpu['backend']}): {gpu['median_ms']:.1f} ms/step (median), "
          f"{gpu['mean_ms']:.1f} mean, {gpu['params']} params")

    speedup = cpu["median_ms"] / gpu["median_ms"] if gpu["median_ms"] else float("nan")
    print(f"\n  GPU speedup: {speedup:.2f}x")

    # project onto the real epoch length (182,377 windows / 128 per batch)
    n_batches = 182377 // BATCH
    print(f"\n  projected epoch ({n_batches} batches):")
    print(f"    CPU: {cpu['median_ms'] * n_batches / 1000 / 60:.1f} min")
    print(f"    GPU: {gpu['median_ms'] * n_batches / 1000 / 60:.1f} min")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
