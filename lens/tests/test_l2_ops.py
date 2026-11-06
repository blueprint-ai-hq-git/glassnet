"""lens/tests/test_l2_ops.py — every op, GPU vs our own CPU kernels.

This is the L2 gate proper. The smoke proved the path works for a few ops; this checks the
whole set the training loop needs, including the two gradients that only appear in backward
(matmul_abt) and the fused loss.

GPU results are compared against OUR OWN CPU kernels, not numpy. Where they disagree, one of our
two implementations is wrong and the difference says which: the CPU kernels are independently
verified against numpy in the L0 gate, so a disagreement here points at the GPU side.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

RNG = np.random.default_rng(0)
RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<32} {detail}")


print("=" * 96)
print("L2 OPS GATE — every op, GPU vs our own CPU kernels")
print("=" * 96)

from lens import _capi as C     # noqa: E402
from lens import cuda as G      # noqa: E402

i = G.info()
print(f"device: {i['device']}  ({i['total_mib']:.0f} MiB)   PTX {i['ptx_bytes']} bytes")
print(f"nvrtc: {i['nvrtc']['version'][0]}.{i['nvrtc']['version'][1]}")

n = 500_000
a = RNG.standard_normal(n).astype(np.float32)
b = RNG.standard_normal(n).astype(np.float32)

# ---------------------------------------------------------------- elementwise
for name, gfn, cfn in [
    ("add", lambda o: G.add(a, b, o), lambda o: C.add(a, b, o)),
    ("mul", lambda o: G.mul(a, b, o), lambda o: C.mul(a, b, o)),
]:
    g = np.zeros(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    gfn(g)
    cfn(c)
    d = float(np.abs(g - c).max())
    report(f"GPU {name} == CPU", d == 0.0, f"max abs diff {d:.3e}")

g = np.zeros(n, dtype=np.float32)
c = np.zeros(n, dtype=np.float32)
G.mul_scalar(a, 2.5, g)
C.mul_scalar(a, 2.5, c)
d = float(np.abs(g - c).max())
report("GPU mul_scalar == CPU", d == 0.0, f"max abs diff {d:.3e}")

for name, gfn, cfn in [
    ("relu", lambda o: G.relu(a, o), lambda o: C.relu(a, o)),
    ("sigmoid", lambda o: G.sigmoid(a, o), lambda o: C.sigmoid(a, o)),
]:
    g = np.zeros(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    gfn(g)
    cfn(c)
    d = float(np.abs(g - c).max())
    report(f"GPU {name} == CPU", d < 1e-6, f"max abs diff {d:.3e}")

# ---------------------------------------------------------------- matmuls
for (m_, k_, n_) in [(64, 784, 128), (128, 128, 128), (256, 256, 256)]:
    A = RNG.standard_normal((m_, k_)).astype(np.float32)
    B = RNG.standard_normal((k_, n_)).astype(np.float32)
    g = np.zeros((m_, n_), dtype=np.float32)
    c = np.zeros((m_, n_), dtype=np.float32)
    G.matmul(A, B, g)
    C.matmul(A, B, c)
    scale = max(float(np.abs(c).max()), 1.0)
    rel = float(np.abs(g - c).max()) / scale
    report(f"GPU matmul {m_}x{k_}x{n_}", rel < 1e-5, f"max rel diff {rel:.3e}")

    B2 = RNG.standard_normal((n_, k_)).astype(np.float32)
    g = np.zeros((m_, n_), dtype=np.float32)
    c = np.zeros((m_, n_), dtype=np.float32)
    G.matmul_abt(A, B2, g)
    C.matmul_abt(A, B2, c)
    scale = max(float(np.abs(c).max()), 1.0)
    rel = float(np.abs(g - c).max()) / scale
    report(f"GPU matmul_abt {m_}x{k_}x{n_}", rel < 1e-5, f"max rel diff {rel:.3e}")

# ---------------------------------------------------------------- fused softmax + CE
rows, V = 512, 151
logits = RNG.standard_normal((rows, V)).astype(np.float32)
labels = RNG.integers(0, V, rows).astype(np.int32)

gp = np.zeros((rows, V), dtype=np.float32)
gd = np.zeros((rows, V), dtype=np.float32)
gloss = G.softmax_ce(logits, labels, gp, gd)

cp = np.zeros((rows, V), dtype=np.float32)
cd = np.zeros((rows, V), dtype=np.float32)
closs = C.softmax_ce(logits, labels, cp, cd)

report("GPU softmax_ce probs == CPU", float(np.abs(gp - cp).max()) < 1e-6,
       f"max abs diff {float(np.abs(gp - cp).max()):.3e}")
report("GPU softmax_ce dlogits == CPU", float(np.abs(gd - cd).max()) < 1e-6,
       f"max abs diff {float(np.abs(gd - cd).max()):.3e}")
report("GPU softmax_ce loss == CPU", abs(gloss - closs) < 1e-5,
       f"GPU {gloss:.6f}  CPU {closs:.6f}  diff {abs(gloss - closs):.3e}")

print("\n" + "=" * 96)
npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
print(f"L2 OPS GATE: {npass}/{len(RESULTS)} passed")
for nm, st, dd in RESULTS:
    if st != "PASS":
        print(f"  !! {nm}: {dd}")
print("=" * 96)
sys.exit(0 if npass == len(RESULTS) else 1)
