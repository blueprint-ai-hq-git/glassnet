#!/usr/bin/env python3
"""gradcheck_width.py - the numerical gradient check, run at a chosen network WIDTH.

Why this exists: the gate battery had only ever been run at one width (d=256), so "the gradients are
correct" was never tested across sizes. This builds a real LensNet at the requested width and checks
EVERY parameter (embedding, concept layer, answer layer) against finite differences, in float64.

The check is a central difference (loss(+eps) - loss(-eps)) / 2eps on each parameter element, compared
against the engine's own backward pass. It reports the worst absolute and relative error and the
tolerance it used, so a PASS is auditable.

Usage: python3 lens/scripts/gradcheck_width.py --width 128 [--vocab 64] [--window 8] [--concepts 6]
"""
import argparse
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
sys.path.insert(0, f"{ROOT}/lens/scripts")
os.chdir(ROOT)

import lens  # noqa: E402
import lens.model as M  # noqa: E402
import lens.tensor as T  # noqa: E402
from lens.concepts import Concept  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--concepts", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--dtype", default="float64",
                    help="the check runs in float64: in float32 the central difference is dominated by "
                         "rounding noise (measured: worst relative error 1.0 at eps=1e-5), which tells "
                         "you nothing about whether the gradient formula is right")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    concepts = []
    for k in range(a.concepts):
        ids = [int(i) for i in rng.choice(a.vocab, size=3, replace=False)]
        concepts.append(Concept(f"gc_concept_{k}", ids))
    net = M.LensNet(a.vocab, concepts, d=a.width, window=a.window, seed=a.seed)
    # cast every parameter to the check's dtype so the comparison is limited by the formula, not by
    # float32 rounding
    # Use the ENGINE's own dtype switch, not a manual cast: the ops build their outputs in the working
    # dtype (float32 by default), so a manually-cast parameter still gets rounded inside the ops and the
    # central difference comes back quantised (measured: relative error exactly 1.0). The engine's
    # docstring says the gradient gate runs in float64 for this reason; this is that switch.
    dt = np.dtype(a.dtype)
    T.set_dtype(dt)
    for prm in net.parameters():
        prm.data = np.asarray(prm.data).astype(dt)
    ids = rng.integers(0, a.vocab, size=(6, a.window)).astype(np.int64)
    y = rng.integers(0, a.vocab, size=6).astype(np.int64)

    def _loss_scalar():
        out = net.loss(ids, y, use_concept_loss=True)
        return out[0] if isinstance(out, tuple) else out

    def total_loss():
        return float(_loss_scalar().data)

    # analytic
    for p in net.parameters():
        p.grad = None
    L = _loss_scalar()
    L.backward()
    analytic = {p.name: np.array(p.grad, dtype=np.float64, copy=True) for p in net.parameters()
                if p.grad is not None}

    worst_abs, worst_rel, checked, missing = 0.0, 0.0, 0, []
    for p in net.parameters():
        if p.name not in analytic:
            missing.append(p.name)
            continue
        arr = p.data if isinstance(p.data, np.ndarray) else np.asarray(p.data)
        flat = arr.reshape(-1)
        ga = analytic[p.name].reshape(-1)
        for i in range(flat.size):
            orig = float(flat[i])
            flat[i] = orig + a.eps
            lp = total_loss()
            flat[i] = orig - a.eps
            lm = total_loss()
            flat[i] = orig
            num = (lp - lm) / (2 * a.eps)
            an = float(ga[i])
            worst_abs = max(worst_abs, abs(num - an))
            denom = max(abs(num) + abs(an), 1e-12)
            worst_rel = max(worst_rel, abs(num - an) / denom)
            checked += 1

    ok = (worst_abs <= a.tol) and not missing
    print(f"  width d={a.width} [{a.dtype}, eps {a.eps:.0e}]: checked {checked} parameter elements across "
          f"{len(analytic)} tensors (vocab {a.vocab}, window {a.window}, {a.concepts} concepts)")
    print(f"    worst absolute error {worst_abs:.3e}   worst relative error {worst_rel:.3e}   "
          f"tolerance {a.tol:.0e}   [{'PASS' if ok else 'FAIL'}]")
    if missing:
        print(f"    tensors with NO gradient (a real defect): {missing}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())