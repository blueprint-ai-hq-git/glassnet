#!/usr/bin/env python3
"""opt_smoke.py - do the SHIPPED callers of RelativeGrad still work after the lens/opt.py change?

The change added a `decay` keyword (default 1.0) and moved the `scale` factor to rmsprop only. Every caller in
the project passes lr= and mode= as keywords (lens/methods.py, lens/scripts/settings_search.py,
lens/scripts/rules_test.py, lens/scripts/reward_hacking_audit.py, lens/stream_train.py), so nothing should
break - this checks that instead of assuming it. Writes a dated block to stdout; the caller appends it to
lens/runs/NUMERIC-PATH-OPT-DECAY.txt.
"""
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
os.chdir(ROOT)

from lens import tensor as T            # noqa: E402
from lens.opt import RelativeGrad       # noqa: E402

CALLS = [
    ("lens/methods.py:167", dict(lr=0.02, mode="relgrad")),
    ("lens/scripts/settings_search.py:71", dict(lr=0.05, mode="relgrad")),
    ("lens/scripts/rules_test.py:141", dict(lr=0.02, mode="relgrad")),
    ("lens/scripts/reward_hacking_audit.py:105", dict(lr=0.02, mode="relgrad")),
    ("lens/stream_train.py:111", dict(lr=0.005, mode="relgrad")),
    ("the new knob, as measured", dict(lr=0.02, mode="relgrad", momentum=0.9, decay=0.999)),
]


def main():
    print("  the shipped callers of RelativeGrad, each with the exact keyword set it uses, stepped 5 times")
    ok = True
    for where, kw in CALLS:
        p = T.Tensor(np.zeros(8, dtype=np.float32), requires_grad=True, name="smoke")
        try:
            opt = RelativeGrad([p], **kw)
            norms = []
            for _ in range(5):
                p.grad = np.ones(8, dtype=np.float32)
                opt.step()
                norms.append(float(np.linalg.norm(np.asarray(p.data, dtype=np.float64))) / max(opt.lr, 1e-12))
            finite = bool(np.isfinite(np.asarray(p.data)).all())
            ok = ok and finite
            print(f"    {where:38s} {str(kw):58s} finite {str(finite):5s} "
                  f"movement after 5 steps / lr = {norms[-1]:.4f}")
        except Exception as exc:                                   # noqa: BLE001
            ok = False
            print(f"    {where:38s} {str(kw):58s} RAISED {type(exc).__name__}: {exc}")
    print(f"  SHIPPED-CALLER SMOKE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
