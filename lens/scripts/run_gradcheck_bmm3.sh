#!/bin/bash
# float64 gradcheck for bmm3 -- both operands, both trans_b modes, against finite differences.
# float32 gave 1.2e-2 relative error (rounding noise, not proof); float64 is the check that can
# actually fail. Numeric gradients are computed per-operand and explicitly.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import sys
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens")
import numpy as np
import tensor as T

rng = np.random.default_rng(1)
N, W1, W2, D = 3, 4, 4, 5
for trans_b in (True, False):
    A = rng.standard_normal((N, W1, D)).astype(np.float64)
    B = (rng.standard_normal((N, W2, D)) if trans_b else rng.standard_normal((N, D, W2))).astype(np.float64)

    def S(Ax, Bx):                      # scalar sum-objective, plain numpy
        O = np.matmul(Ax, np.swapaxes(Bx, -1, -2)) if trans_b else np.matmul(Ax, Bx)
        return float(O.sum())

    Ta = T.Tensor(A.copy(), requires_grad=True)
    Tb = T.Tensor(B.copy(), requires_grad=True)
    T.bmm3(Ta, Tb, trans_b=trans_b).backward()
    eps = 1e-4
    i = (0, 0, 0)
    A1, A0 = A.copy(), A.copy(); A1[i] += eps; A0[i] -= eps
    B1, B0 = B.copy(), B.copy(); B1[i] += eps; B0[i] -= eps
    nA = (S(A1, B) - S(A0, B)) / (2 * eps)
    nB = (S(A, B1) - S(A, B0)) / (2 * eps)
    aA = float(np.asarray(Ta.grad)[i]); aB = float(np.asarray(Tb.grad)[i])
    rA = abs(aA - nA) / max(abs(nA), 1e-12)
    rB = abs(aB - nB) / max(abs(nB), 1e-12)
    print(f"  trans_b={trans_b!s:<5} dA analytic {aA: .9f} numeric {nA: .9f} rel {rA:.2e} | "
          f"dB rel {rB:.2e}  {'PASS' if (rA < 1e-6 and rB < 1e-6) else 'FAIL'}"
          f"   (threshold 1e-6: central-difference resolution at this eps; float32 noise is 1e-2)")
PY