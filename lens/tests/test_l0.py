"""lens/tests/test_l0.py — THE L0 GATE.

Every kernel in liblens.so is checked against NumPy on a fixed random battery. NumPy is used
here as a REFERENCE ORACLE only -- it computes the expected answer, our kernels compute the
actual one, and the difference is printed. A kernel that disagrees is a bug in OUR code.

Tolerance is stated, not hidden: -ffast-math lets the compiler reassociate float adds, so the
results are not bit-identical to NumPy. The bar is 1e-4 relative on values of order 1, which is
far tighter than any training run could notice and far looser than the reassociation error.

Also measured: speed against NumPy's BLAS, reported whatever it is.
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from lens import _capi as C  # noqa: E402

RNG = np.random.default_rng(0)
RESULTS = []
TOL = 1e-4


def check(name, ours, ref):
    ref = np.asarray(ref, dtype=np.float64)
    ours = np.asarray(ours, dtype=np.float64)
    if ours.shape != ref.shape:
        RESULTS.append((name, "FAIL", f"shape {ours.shape} != {ref.shape}"))
        print(f"  FAIL  {name}: shape {ours.shape} != {ref.shape}")
        return
    scale = max(float(np.abs(ref).max()), 1.0)
    absd = float(np.abs(ours - ref).max())
    rel = absd / scale
    ok = rel <= TOL
    RESULTS.append((name, "PASS" if ok else "FAIL", f"max abs {absd:.3e} rel {rel:.3e}"))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<28} max abs diff {absd:.3e}  rel {rel:.3e}")


print("=" * 96)
print("L0 GATE — our own kernels vs NumPy as the reference oracle")
print("=" * 96)
print(f"tolerance: {TOL:g} relative (stated, because -ffast-math allows reassociation)")

# ---------------------------------------------------------------- elementwise
n = 100_000
a = RNG.standard_normal(n).astype(np.float32)
b = (RNG.standard_normal(n).astype(np.float32) + 2.0)

out = C.f32(n)
C.add(a, b, out);      check("add", out, a + b)
C.sub(a, b, out);      check("sub", out, a - b)
C.mul(a, b, out);      check("mul", out, a * b)
C.div(a, b, out);      check("div", out, a / b)
C.add_scalar(a, 2.5, out); check("add_scalar", out, a + 2.5)
C.mul_scalar(a, -1.5, out); check("mul_scalar", out, a * -1.5)
C.relu(a, out);        check("relu", out, np.maximum(a, 0))
C.sigmoid(a, out);     check("sigmoid", out, 1.0 / (1.0 + np.exp(-a.astype(np.float64))))
check("sum", np.array([C.total(a)]), np.array([a.astype(np.float64).sum()]))

# ---------------------------------------------------------------- matmul
for (m, k, nn) in [(64, 784, 128), (128, 128, 128), (512, 256, 151)]:
    A = RNG.standard_normal((m, k)).astype(np.float32)
    B = RNG.standard_normal((k, nn)).astype(np.float32)
    O = C.f32((m, nn))
    C.matmul(A, B, O)
    check(f"matmul {m}x{k}x{nn}", O, A.astype(np.float64) @ B.astype(np.float64))
    A2 = RNG.standard_normal((k, m)).astype(np.float32)
    O2 = C.f32((m, nn))
    C.matmul_atb(A2, B, O2)
    check(f"matmul_atb {m}x{k}x{nn}", O2, A2.astype(np.float64).T @ B.astype(np.float64))
    B2 = RNG.standard_normal((nn, k)).astype(np.float32)
    O3 = C.f32((m, nn))
    C.matmul_abt(A, B2, O3)
    check(f"matmul_abt {m}x{k}x{nn}", O3, A.astype(np.float64) @ B2.astype(np.float64).T)

# ---------------------------------------------------------------- fused softmax + CE
rows, V = 256, 151
logits = RNG.standard_normal((rows, V)).astype(np.float32)
labels = RNG.integers(0, V, rows).astype(np.int32)
probs = C.f32((rows, V))
dlog = C.f32((rows, V))
loss = C.softmax_ce(logits, labels, probs, dlog)

z = logits.astype(np.float64)
z = z - z.max(1, keepdims=True)
e = np.exp(z)
ref_p = e / e.sum(1, keepdims=True)
ref_loss = float(-np.log(ref_p[np.arange(rows), labels]).mean())
ref_d = ref_p.copy()
ref_d[np.arange(rows), labels] -= 1.0
check("softmax_ce loss", np.array([loss]), np.array([ref_loss]))
check("softmax_ce probs", probs, ref_p)
check("softmax_ce dlogits", dlog, ref_d)

# ---------------------------------------------------------------- speed
print("\n" + "-" * 96)
print("SPEED — ours vs NumPy's BLAS, reported whatever it is")
print("-" * 96)
m, k, nn = 512, 784, 256
A = RNG.standard_normal((m, k)).astype(np.float32)
B = RNG.standard_normal((k, nn)).astype(np.float32)
O = C.f32((m, nn))

C.matmul(A, B, O)
t0 = time.perf_counter()
for _ in range(5):
    C.matmul(A, B, O)
t_ours = (time.perf_counter() - t0) / 5

A @ B
t0 = time.perf_counter()
for _ in range(5):
    _ = A @ B
t_np = (time.perf_counter() - t0) / 5

flops = 2.0 * m * k * nn
print(f"  matmul {m}x{k}x{nn}:  ours {t_ours*1e3:8.2f} ms ({flops/t_ours/1e9:6.2f} GFLOPS)   "
      f"numpy {t_np*1e3:8.2f} ms ({flops/t_np/1e9:6.2f} GFLOPS)   ratio {t_ours/t_np:5.2f}x")

n = 5_000_000
a = RNG.standard_normal(n).astype(np.float32)
b = RNG.standard_normal(n).astype(np.float32)
out = C.f32(n)
C.add(a, b, out)
t0 = time.perf_counter()
for _ in range(20):
    C.add(a, b, out)
t_ours = (time.perf_counter() - t0) / 20
_ = a + b
t0 = time.perf_counter()
for _ in range(20):
    _ = a + b
t_np = (time.perf_counter() - t0) / 20
print(f"  add {n} elems:      ours {t_ours*1e3:8.2f} ms   numpy {t_np*1e3:8.2f} ms   "
      f"ratio {t_ours/t_np:5.2f}x")

print("\n" + "=" * 96)
npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
print(f"L0 GATE: {npass}/{len(RESULTS)} passed")
for nm, st, d in RESULTS:
    if st != "PASS":
        print(f"  !! {nm}: {d}")
print("=" * 96)
sys.exit(0 if npass == len(RESULTS) else 1)
