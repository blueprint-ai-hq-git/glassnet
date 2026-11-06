"""lens/tests/test_l2_train.py — does the MODEL train on the GPU, and identically?

The right gate for a device switch is not "it runs", it is "it gives the SAME NUMBERS". Our GPU
kernels are exact for matmul (0.000e+00 in the ops gate) and ~1e-7 for sigmoid and the fused
loss, so a training run on the GPU must track the CPU run closely. A large divergence means the
device switch is wrong, not that floating point is noisy.

Also measured: the speed, reported whatever it is.

Device policy (stated, not glossed): matmul, matmul_abt and the fused softmax+CE go to the GPU --
those are the compute-bound ops and about 90% of the FLOPs. Elementwise ops stay on the CPU,
because an elementwise add is memory-bound and shipping buffers to the card and back would cost
more than the add itself.
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

RNG = np.random.default_rng(0)
RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<40} {detail}")


print("=" * 100)
print("L2 TRAIN GATE — the readable model, CPU vs GPU, same numbers")
print("=" * 100)

from lens import backend as B                              # noqa: E402
from lens import tensor as T                              # noqa: E402
from lens.concepts import Concept, explain, coverage       # noqa: E402
from lens.model import LensNet                             # noqa: E402

V, W, N = 24, 6, 6000
X = RNG.integers(0, 12, size=(N, W)).astype(np.int64)
X[: N // 2, 0] = 20
X[N // 2:, 0] = 21
X[::3, 1] = 9
y = ((X[:, 0] == 20) * 4 + (X[:, 1] == 9) * 6 + X[:, 2] % 5).astype(np.int64) % V

concepts = [
    Concept("marker_twenty", [20], support=float((X[:, 0] == 20).mean())),
    Concept("marker_twentyone", [21], support=float((X[:, 0] == 21).mean())),
    Concept("nine_second", [9], support=float((X[:, 1] == 9).mean())),
]


def run(dev, steps=120, batch=128, lr=0.5):
    T.set_device(dev)
    net = LensNet(V, concepts, d=32, window=W, seed=7, lam_concept=0.1, lam_act=0.0)
    rng = np.random.default_rng(1234)
    losses = []
    B.sync()                                  # nothing outstanding before the clock starts
    t0 = time.perf_counter()
    for _ in range(steps):
        for p in net.parameters():
            p.grad = None
        idx = rng.permutation(N)[:batch]
        loss, _ = net.loss(X[idx], y[idx])
        loss.backward()
        for p in net.parameters():
            if p.grad is not None:
                p.data = p.data - lr * p.grad
        losses.append(float(B.to_cpu(loss.data)))
    B.sync()                                  # wait for the GPU before stopping the clock
    dt = time.perf_counter() - t0
    h = net.forward_arrays(X[:600])["concepts"]
    return net, losses, dt, h


print("training 120 steps on CPU...")
net_c, loss_c, t_c, h_c = run("cpu")
print("training 120 steps on GPU...")
net_g, loss_g, t_g, h_g = run("gpu")

report("both devices train", len(loss_c) == len(loss_g) == 120,
       f"final loss CPU {loss_c[-1]:.6f}   GPU {loss_g[-1]:.6f}")

dl = max(abs(a - b) for a, b in zip(loss_c, loss_g))
report("loss curves match", dl < 1e-3,
       f"max |loss_cpu - loss_gpu| over 120 steps = {dl:.3e}")

# parameters -- compared on the HOST, since on the GPU they are cupy arrays and numpy.abs()
# cannot mix with them
worst_p = 0.0
worst_name = ""
for p, q in zip(net_c.parameters(), net_g.parameters()):
    d = float(np.abs(B.to_cpu(p.data) - B.to_cpu(q.data)).max())
    if d > worst_p:
        worst_p, worst_name = d, p.name
report("trained parameters match", worst_p < 1e-4,
       f"worst |cpu - gpu| = {worst_p:.3e} on {worst_name}")

dh = float(np.abs(h_c - h_g).max())
report("concept activations match", dh < 1e-4, f"max |cpu - gpu| = {dh:.3e}")

# ---------------------------------------------------------------- both remain readable
# The device must be set per model: it is PROCESS-GLOBAL, and a model's parameters live on
# whichever device was active when it was built. Explaining a CPU model while the device is
# "gpu" mixes numpy and cupy arrays and cupy refuses -- which is how this constraint was found.
for label, net, dev in (("CPU", net_c, "cpu"), ("GPU", net_g, "gpu")):
    T.set_device(dev)
    e = explain(net, X[0], top_k=2)
    report(f"{label}: explain() still EXACT", e["residual"] < 1e-4,
           f"residual {e['residual']:.3e}")
    cov = coverage(net, X[:600])
    report(f"{label}: coverage 100%", abs(cov - 1.0) < 1e-9, f"coverage {cov*100:.4f}%")

# ---------------------------------------------------------------- speed
print("\n" + "-" * 100)
print("SPEED — 120 training steps, reported whatever it is")
print("-" * 100)
print(f"  CPU {t_c:7.3f} s   GPU {t_g:7.3f} s   speedup {t_c/t_g:5.2f}x")
report("GPU training is faster", t_g < t_c, f"{t_c/t_g:.2f}x ({t_c:.2f}s -> {t_g:.2f}s)")

T.set_device("cpu")
print("\n" + "=" * 100)
npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
print(f"L2 TRAIN GATE: {npass}/{len(RESULTS)} passed")
for nm, st, dd in RESULTS:
    if st != "PASS":
        print(f"  !! {nm}: {dd}")
print("=" * 100)
sys.exit(0 if npass == len(RESULTS) else 1)
