#!/usr/bin/env python3
"""f16_gemv_smoke.py - pre-flight for the QUEUED (f2b) job: one ship, three steps, no timing.

It exists so the queued job is never the first thing that runs this code. It answers only: does the
kernel load, is it called, does it produce the same tokens as the float32 path on a short walk, and does
the CONTROL (column-reversed answer.W) actually change them? Prints to stdout; writes nothing.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import json                      # noqa: E402
import sys                       # noqa: E402

import numpy as np               # noqa: E402

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
for _p in (ROOT, ROOT + "/lens/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lens                                                      # noqa: E402
from lens.kvcache import WindowCache, cached_logits              # noqa: E402
from lens.fusedgemv import FusedF16WindowCache, gemv_singleton   # noqa: E402

SHIP = os.path.join(ROOT, "lens/runs/chat/cl_d256_attn_12_ep12.npz")
CORPUS = os.path.join(ROOT, "data/chat/prepared_mix_ctrl_honest/vocab_v12000.json")

g = gemv_singleton()
print(f"has_avx2 {g.has_avx2}   build {' '.join(g.cmd)}")

net = lens.load(SHIP, device="cpu")
vocab = json.load(open(CORPUS))
if net.V == len(vocab) + 2:
    vocab = vocab + ["<|im_start|>", "<|im_end|>"]
vmap = {w: i for i, w in enumerate(vocab)}
import re                                                        # noqa: E402
enc = [vmap.get(w, 0) for w in re.findall(r"\w+|[^\w\s]", "hello , how are you ?")]
enc = [0] * (net.W - len(enc)) + enc
enc = enc[-net.W:]
print(f"ship d={net.D} W={net.W} K={net.K} V={net.V}")

for label, cache in (("f16gemv", FusedF16WindowCache(net, nthreads=3).seed(enc)),
                     ("control", FusedF16WindowCache(net, nthreads=3, control=True).seed(enc))):
    ref = WindowCache(net, fused=True).seed(enc)
    worst, rt, qt = 0.0, [], []
    for _ in range(3):
        zr = np.asarray(cached_logits(net, ref), dtype=np.float64)
        zq = np.asarray(cache.logits(), dtype=np.float64)
        worst = max(worst, float(np.abs(zr - zq).max()))
        nxt = int(np.argmax(zr))
        rt.append(nxt)
        qt.append(int(np.argmax(zq)))
        ref.push(nxt)
        cache.push(nxt)
    print(f"{label:8s} kernel calls {cache.kernel_calls}  max|dz| {worst:.6e}  "
          f"ref tokens {rt}  this path {qt}  identical {rt == qt}")
print("SMOKE DONE")
