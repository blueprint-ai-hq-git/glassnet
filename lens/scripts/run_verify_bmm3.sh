#!/bin/bash
# VERIFY the bmm3 speed fix BEFORE trusting it: (1) forward outputs identical to the einsum
# version, (2) gradients identical through the engine's own autograd, (3) timing on GPU
# (batched matmul vs einsum), (4) the engine's own gradcheck-style numeric check on bmm3.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import sys, time
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens")
import numpy as np
import tensor as T
import backend

rng = np.random.default_rng(0)
N, W, D = 64, 32, 512
for dev in ("cpu", "gpu"):
    try:
        backend.use_gpu(dev == "gpu")
    except Exception as e:
        print(f"  [{dev}] unavailable: {e}"); continue
    xp = backend.xp
    a = xp.asarray(rng.standard_normal((N, W, D)).astype(np.float32))
    b = xp.asarray(rng.standard_normal((N, W, D)).astype(np.float32))
    g = xp.asarray(rng.standard_normal((N, W, W)).astype(np.float32))
    # 1) forward equality (trans_b = the attention scores path)
    ref = xp.einsum("nij,nkj->nik", a, b)
    new = xp.matmul(a, xp.swapaxes(b, -1, -2))
    d1 = float(xp.max(xp.abs(ref - new)))
    # 2) backward equality against the einsum formulas
    ga_ref = xp.einsum("nab,nbj->naj", g, b); ga_new = xp.matmul(g, b)
    gb_ref = xp.einsum("nab,naj->nbj", g, a); gb_new = xp.matmul(xp.swapaxes(g, -1, -2), a)
    d2 = float(xp.max(xp.abs(ga_ref - ga_new))); d3 = float(xp.max(xp.abs(gb_ref - gb_new)))
    # 3) through the engine itself (autograd), both variants must agree with finite differences
    # engine-side check: d(sum of out)/dA element 0,0,0  vs finite differences of the same sum
    A0 = np.asarray(a).copy(); B0 = np.asarray(b).copy()
    Ta = T.Tensor(A0, requires_grad=True); Tb = T.Tensor(B0, requires_grad=True)
    out = T.bmm3(Ta, Tb, trans_b=True)
    out.backward()                      # engine convention: backward() seeds grad = ones (sum)
    eps = 1e-3
    i = (0, 0, 0)
    orig = float(A0[i])
    ap = A0.copy(); am = A0.copy(); ap[i] = orig + eps; am[i] = orig - eps
    Sp = float(np.sum(np.matmul(ap, np.swapaxes(B0, -1, -2))))
    Sm = float(np.sum(np.matmul(am, np.swapaxes(B0, -1, -2))))
    num = (Sp - Sm) / (2 * eps); ana = float(np.asarray(Ta.grad)[i])
    d4 = abs(num - ana) / max(abs(num), abs(ana), 1e-6)
    # 4) timing
    def bench(fn, n=5):
        fn(); backend.sync()
        t0 = time.perf_counter()
        for _ in range(n): fn()
        backend.sync()
        return (time.perf_counter() - t0) / n * 1000
    t_ein = bench(lambda: xp.einsum("nij,nkj->nik", a, b))
    t_mm  = bench(lambda: xp.matmul(a, xp.swapaxes(b, -1, -2)))
    t_ein_bw = bench(lambda: xp.einsum("nab,naj->nbj", g, a))
    t_mm_bw  = bench(lambda: xp.matmul(xp.swapaxes(g, -1, -2), a))
    print(f"  [{dev}] fwd diff {d1:.3e}  dA diff {d2:.3e}  dB diff {d3:.3e}  "
          f"gradcheck rel {d4:.2e}")
    print(f"  [{dev}] scores: einsum {t_ein:8.2f} ms -> matmul {t_mm:8.2f} ms  "
          f"({t_ein/max(t_mm,1e-9):5.1f}x)   grad: {t_ein_bw:8.2f} -> {t_mm_bw:8.2f} ms "
          f"({t_ein_bw/max(t_mm_bw,1e-9):5.1f}x)")
PY