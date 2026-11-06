"""decorr_gpu_smoke.py — prove the decorr penalty survives a backward pass ON THE GPU.

The first version computed the penalty with numpy while the activations lived on the card, so the
backward died mid-run (both K21k runs). This smoke does one real training step on the GPU with the
penalty switched on and prints the value before/after -- if it runs and the value moves, the
device path is real.
"""
import os
import sys

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, P)
sys.path.insert(0, os.path.join(P, "lens", "scripts"))

import lens.tensor as T                              # noqa: E402
from lens import backend as B                        # noqa: E402
from decorr import decorr_term                       # noqa: E402


def main():
    T.set_device("gpu")
    T.set_dtype(np.float32)
    rng = np.random.default_rng(0)
    N, K = 64, 19
    W = T.Tensor(B.to_dev((rng.standard_normal((K, K)) * 0.3).astype(np.float32)),
                 requires_grad=True, name="W")
    X = T.Tensor(B.to_dev(rng.standard_normal((N, K)).astype(np.float32)), requires_grad=True,
                 name="X")
    print("device:", T.device())

    def step(lr=0.1):
        for p in (W, X):
            p.grad = None
        h = T.sigmoid(T.matmul(X, W))
        node, val = decorr_term(h)
        node.backward()
        for p in (W, X):
            if p.grad is not None:
                p.data = p.data - lr * p.grad
        B.sync()
        return val

    v0 = step()
    for _ in range(5):
        v1 = step()
    print(f"decorr penalty on GPU: {v0:.6f} -> {v1:.6f}")
    ok = np.isfinite(v1) and v1 < v0
    print("GPU PATH:", "PASS (the penalty went down under a real backward+step)" if ok
          else "FAIL (check the device handling)")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())