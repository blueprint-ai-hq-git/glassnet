#!/usr/bin/env python3
"""dtype_equiv_check.py - P64c: does the REAL engine give bit-identical numbers for an int32 window array?

Written 2026-10-22 16:4x for prediction P64c (lens/runs/MEMFIX2-PREDICTIONS.txt, written BEFORE the edit).
The width fix changes the window array from int64 to int32. That is only safe if the engine treats the two
identically, so this runs the SAME model on the SAME data both ways and compares, BIT FOR BIT:
    * the logits of a forward pass,
    * the total loss,
    * every parameter's gradient after backward.
Bit-identical is the bar on purpose: "close enough" would not tell us whether the change touched the run.

It is run on BOTH devices (cpu and gpu) because the embedding lookup is the one place the dtype could matter:
lens/tensor.py::embedding indexes the embedding matrix with the ndarray, and cupy's fancy indexing is a
different code path from numpy's. The model here is tiny (d=64, window=32, V=12,002) so a GPU run costs
almost no VRAM - well inside the 5 GB rule.

The model is built exactly like a real one: api.Net, the tied direct path ON (the arm this serves),
K concepts, real vocabulary size. A SKIPPED device is reported as SKIPPED, never as PASS.

Run:  .venv-gpu/bin/python3 -u lens/scripts/dtype_equiv_check.py
Writes nothing but stdout.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lens import api, backend as B                                   # noqa: E402
from lens.concepts import Concept                                    # noqa: E402

V = 12002
W = 32
N = 64


def concepts_for(vocab):
    return [Concept("is_a_question", [11, 12], support=0.10),
            Concept("mentions_a_name", [13, 14, 15], support=0.08),
            Concept("mentions_a_number", [16, 17, 18, 19], support=0.12),
            Concept("mentions_time", [20, 21], support=0.07),
            Concept("is_imperative", [22, 23], support=0.09),
            Concept("mentions_feeling", [24], support=0.05),
            Concept("is_a_list", [25, 26], support=0.06),
            Concept("mentions_place", [27, 28], support=0.03)]


def build(device):
    # api.Net -> LensNet(vocab_size, ...): the REAL vocabulary SIZE (12,002), not a list, exactly as the
    # shipped trainer passes it.
    return api.Net(V, concepts_for(None), d=64, window=W, seed=0, lam_concept=0.3,
                   lam_act=0.0, device=device, direct=True, direct_mode="tied")


def grads_cpu(net):
    return [None if p.grad is None else np.asarray(B.to_cpu(p.grad), dtype=np.float64).copy()
            for p in net.parameters()]


def one_pass(net, X, Y, Wt):
    for p in net.parameters():
        p.grad = None
    total, _parts = net.loss(X, Y, weights=Wt)
    total.backward()
    logits = np.asarray(net.forward_arrays(X)["logits"], dtype=np.float64)
    return float(B.to_cpu(total.data)), logits, grads_cpu(net)


def run_device(device):
    try:
        net = build(device)
    except Exception as e:                                            # noqa: BLE001
        return f"  device {device}: SKIPPED (could not build: {type(e).__name__}: {e})"
    rng = np.random.default_rng(0)
    Xi = rng.integers(0, V, size=(N, W))
    Yi = rng.integers(0, V, size=N)
    Wt = (rng.random(N) > 0.5).astype(np.float32)
    X64, Y64 = Xi.astype(np.int64), Yi.astype(np.int64)
    X32, Y32 = Xi.astype(np.int32), Yi.astype(np.int32)
    l64, lg64, g64 = one_pass(net, X64, Y64, Wt)
    l32, lg32, g32 = one_pass(net, X32, Y32, Wt)
    logits_same = np.array_equal(lg64, lg32)
    loss_same = (l64 == l32)
    ndiff = sum(1 for a, b in zip(g64, g32) if (a is None) != (b is None) or
                (a is not None and not np.array_equal(a, b)))
    n_g = sum(1 for a in g64 if a is not None)
    lines = [f"  device {device}:",
             f"    forward logits  bit-identical: {logits_same}  "
             f"(max |dlogit| {np.abs(lg64 - lg32).max():.3e})",
             f"    total loss      {l64!r}  vs  {l32!r}   identical: {loss_same}",
             f"    gradients       {n_g - ndiff}/{n_g} tensors bit-identical "
             f"({'PASS' if ndiff == 0 else 'FAIL: ' + str(ndiff) + ' differ'})"]
    ok = logits_same and loss_same and ndiff == 0
    return "\n".join(lines) + f"\n    -> {'PASS' if ok else 'FAIL'}", ok


def main():
    print("D-TYPE-EQUIV: int32 vs int64 window arrays through the real engine (tied direct path ON, d=64)")
    print(f"data: {N} windows of {W} token ids, real vocabulary size {V}, real loss with sample weights\n")
    ok_all, seen = True, 0
    for device in ("cpu", "gpu"):
        r = run_device(device)
        if isinstance(r, tuple):
            text, ok = r
            seen += 1
            ok_all = ok_all and ok
        else:
            text, ok = r, True
        print(text)
        print()
    if seen == 0:
        print("P64c INCONCLUSIVE: no device produced a comparison at all.")
        return 2
    print(f"P64c {'PASS' if ok_all else 'FAIL'}: int32 and int64 window arrays are interchangeable in the "
          f"real engine on {seen} device(s) tested.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
