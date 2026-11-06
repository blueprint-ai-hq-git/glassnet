#!/usr/bin/env python3
"""gpu_attn_cache_probe.py - can the server serve an ATTENTION ship on the GPU?

Found while working PLAN item (f): WindowCache reads the ship's weights with `net.<x>.data`, which is a
cupy array when the ship is loaded with device="gpu", but keeps numpy buffers and a numpy causal mask.
That combination is expected to raise, which would mean `lens/serve.py <attention ship> --device gpu`
cannot start a generation at all. A suspicion is not a finding, so this probe runs it and prints the
result. Output: lens/runs/GPU-ATTN-CACHE.txt (appended).
"""
import os
import sys
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import numpy as np                                              # noqa: E402

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
# REMOVED 2026-10-20, THE SAME REAL BUG IN A FORM THE FIRST GATE COULD NOT SEE (class 3;
# evidence lens/runs/SYSPATH-SHADOW-FIX-class3.txt, probe lens/runs/GPU-ATTN-CACHE.txt): this
# tuple put <project>/lens on sys.path as well, so the top-level name `cuda` resolved to our own
# lens/cuda/ package and cupy 14.2 died with "ModuleNotFoundError: No module named
# 'cuda.pathfinder'". lens/ is imported as `lens.X` via the ROOT entry and lens/scripts/ has its
# own entry, so the `/lens` element is not needed. Do not add it back.
for _p in (ROOT, ROOT + "/lens/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lens                                                     # noqa: E402
from lens.decode import make_cache, step_logits                 # noqa: E402

SHIP = "lens/runs/chat/cl_d256_attn_12_ep12.npz"


def main():
    print(f"probe started {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")
    for device in ("cpu", "gpu"):
        print(f"\n--- device={device}")
        try:
            net = lens.load(os.path.join(ROOT, SHIP), device=device)
        except Exception as e:
            print(f"    ship would not load on {device}: {type(e).__name__}: {e}")
            continue
        print(f"    attn={'yes' if net.attn is not None else 'no'}  emb dtype/type="
              f"{type(net.emb.data).__name__}")
        ctx = [1] * int(net.W)
        try:
            cache, path = make_cache(net, ctx)
            print(f"    make_cache -> {path!r}")
            z = step_logits(net, ctx, cache)
            print(f"    step_logits OK, logits shape {np.asarray(z).shape}, "
                  f"argmax {int(np.argmax(np.asarray(z)))}")
            if cache is not None:
                cache.push(7)
                z2 = step_logits(net, ctx, cache)
                print(f"    push+step OK, argmax {int(np.argmax(np.asarray(z2)))}")
        except Exception as e:
            print(f"    RAISED {type(e).__name__}: {e}")
            tb = traceback.format_exc().strip().splitlines()
            for line in tb[-6:]:
                print(f"      {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
