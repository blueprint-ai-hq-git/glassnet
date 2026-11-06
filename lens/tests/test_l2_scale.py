"""lens/tests/test_l2_scale.py — where does the GPU actually start winning?

The CPU-vs-GPU gate says the GPU is 0.24x at d=32, K=3, batch=128 -- four times SLOWER. That is
not a surprise once it is measured: a tiny model issues roughly 200 kernel launches per step and
each launch costs ~15 microseconds regardless of how little work it does. 120 steps x 200 x 15us
is about 0.36 s, which is the entire measured GPU time.

So the question is not "is the GPU faster" but "at what size does it start being faster". That is
a measurement, not an opinion. This trains the SAME architecture at increasing sizes on both
devices and reports the crossover honestly -- including if there is no crossover in the range.
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

print("=" * 100)
print("WHERE DOES THE GPU START WINNING? — same architecture, increasing size")
print("=" * 100)

from lens import backend as B                              # noqa: E402
from lens import tensor as T                              # noqa: E402
from lens.concepts import Concept                          # noqa: E402
from lens.model import LensNet                             # noqa: E402

# (vocab, d, batch, window) -- a model small enough to be "toy" up to one with real matmuls
SIZES = [
    ("toy",    50,  32,  128, 6),
    ("small",  200, 128, 512, 8),
    ("medium", 1000, 256, 1024, 8),
]

STEPS = 40
rows = []

for label, V, D, BATCH, W in SIZES:
    N = BATCH * 8
    rng = np.random.default_rng(0)
    X = rng.integers(0, V, size=(N, W)).astype(np.int64)
    y = rng.integers(0, V, size=N).astype(np.int64)
    concepts = [Concept(f"tok_{i}", [i + 1], support=1.0 / V) for i in range(3)]

    params = 0
    times = {}
    for dev in ("cpu", "gpu"):
        T.set_device(dev)
        net = LensNet(V, concepts, d=D, window=W, seed=3, lam_concept=0.1, lam_act=0.0)
        params = sum(p.data.size for p in net.parameters())
        opt_rng = np.random.default_rng(7)
        B.sync()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            for p in net.parameters():
                p.grad = None
            idx = opt_rng.permutation(N)[:BATCH]
            loss, _ = net.loss(X[idx], y[idx])
            loss.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.data = p.data - 0.3 * p.grad
        B.sync()
        times[dev] = time.perf_counter() - t0
    rows.append((label, V, D, BATCH, params, times["cpu"], times["gpu"]))
    print(f"  {label:<7} V={V:<5} d={D:<4} batch={BATCH:<5} {params:>9,} params   "
          f"CPU {times['cpu']:.3f}s   GPU {times['gpu']:.3f}s   "
          f"{times['cpu']/times['gpu']:.2f}x")

T.set_device("cpu")

print("\n" + "-" * 100)
print("SUMMARY")
print("-" * 100)
print(f"  {'size':<8}{'params':>12}{'CPU s':>10}{'GPU s':>10}{'speedup':>10}   verdict")
for label, V, D, BATCH, params, tc, tg in rows:
    sp = tc / tg
    verdict = "GPU wins" if sp > 1.05 else ("CPU wins" if sp < 0.95 else "tie")
    print(f"  {label:<8}{params:>12,}{tc:>10.3f}{tg:>10.3f}{sp:>9.2f}x   {verdict}")

crossed = [r for r in rows if r[5] / r[6] > 1.0]
print()
if crossed:
    first = crossed[0]
    print(f"  CROSSOVER: the GPU starts winning at {first[0]} "
          f"({first[4]:,} params, batch {first[3]}).")
else:
    print("  NO CROSSOVER in the sizes tested -- the CPU wins at every size here.")
print("  Reported as measured. Launch overhead dominates below the crossover; that is a real")
print("  property of small models on a GPU, not a defect in the kernel.")
