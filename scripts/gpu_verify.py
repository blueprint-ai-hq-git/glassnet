"""GPU verification — run with the cupy backend.

TWO checks, in this order:
  1. GATE 1 on the GPU: the same 9 hand-written gradient checks that pass on
     CPU, re-run with the cupy backend. If the GPU backward is wrong, the
     gate fails and no GPU number may be used.
  2. CPU-vs-GPU PARITY: identical net, seed, data and step count on both
     backends. PRE-REGISTERED tolerances (written before running):
       - step-0 loss must agree to a relative 1e-5 (same initial weights)
       - final loss must agree to a relative 2e-2 (float32, different
         reduction orders on the GPU, amplified over 60 optimiser steps)
       - argmax predictions must agree on >= 99% of samples
     If parity fails, GPU results are NOT comparable to the CPU table and
     must not be mixed into any comparison group.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from glassnet import tensor as T
from glassnet.tensor import Tensor, set_backend, backend_name
from glassnet.layers import Dense, Sequential
from glassnet.losses import cross_entropy
from glassnet.optim import SGD


def parity_run(backend, seed=0, steps=60, n=512, d=32, c=10, lr=0.1):
    set_backend(backend)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, c, size=n)
    net = Sequential(
        Dense(d, 24, rng=np.random.default_rng(seed + 1), name="l1"), T.relu,
        Dense(24, c, rng=np.random.default_rng(seed + 2), name="l2"))
    opt = SGD(net.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = cross_entropy(net(Tensor(X)), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    pred = net(Tensor(X)).data
    if backend == "cupy":
        pred = pred.get()          # cupy forbids implicit host conversion
    return np.array(losses), pred.argmax(axis=1)


def main():
    print("=" * 66)
    print("GPU VERIFICATION")
    print("=" * 66)

    # ---- 1. gate on the GPU -------------------------------------------
    print("\n--- 1. GATE 1 on the GPU (cupy backend) ---")
    set_backend("cupy")
    print("backend:", backend_name())
    try:
        from glassnet.gates import run_gate1
        ok_gpu = run_gate1(verbose=True)
    except Exception as e:
        print("GPU GATE FAILED TO RUN:", type(e).__name__, e)
        ok_gpu = False
    print("GATE 1 on GPU:", "PASS" if ok_gpu else "FAIL")

    # ---- 2. parity -----------------------------------------------------
    print("\n--- 2. CPU vs GPU parity (same seed/data/steps) ---")
    t0 = time.time(); cpu_loss, cpu_pred = parity_run("numpy"); t_cpu = time.time() - t0
    t0 = time.time(); gpu_loss, gpu_pred = parity_run("cupy"); t_gpu = time.time() - t0

    l0_rel = abs(cpu_loss[0] - gpu_loss[0]) / (abs(cpu_loss[0]) + 1e-12)
    lf_rel = abs(cpu_loss[-1] - gpu_loss[-1]) / (abs(cpu_loss[-1]) + 1e-12)
    agree = float((cpu_pred == gpu_pred).mean())

    print(f"  step-0 loss   cpu {cpu_loss[0]:.8f}   gpu {gpu_loss[0]:.8f}   rel {l0_rel:.2e}")
    print(f"  final  loss   cpu {cpu_loss[-1]:.8f}   gpu {gpu_loss[-1]:.8f}   rel {lf_rel:.2e}")
    print(f"  argmax agreement: {agree*100:.2f}%")
    print(f"  60-step wall time: cpu {t_cpu:.2f}s   gpu {t_gpu:.2f}s")

    p1 = l0_rel < 1e-5
    p2 = lf_rel < 2e-2
    p3 = agree >= 0.99
    print(f"  pre-registered: step0<1e-5 {p1} | final<2e-2 {p2} | agree>=99% {p3}")
    parity_ok = p1 and p2 and p3
    print("PARITY:", "PASS" if parity_ok else "FAIL")

    print("\n" + "=" * 66)
    print(f"VERDICT: gate1_gpu={ok_gpu}  parity={parity_ok}")
    print("=" * 66)
    return 0 if (ok_gpu and parity_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
