#!/usr/bin/env python3
"""Probe: what is inside the clean ship and what does the tokenizer/corpus look like."""
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SHIP = "lens/runs/chat/cl_d256_clean_24_ep24.npz"
z = np.load(SHIP, allow_pickle=True)
print("arrays:", len(z.files))
for k in z.files:
    a = z[k]
    print(f"  {k:24s} {getattr(a, 'shape', None)} {a.dtype}")

import lens.api as api
net = api.load(SHIP, device="cpu")
print("net: D", net.D, "W", net.W, "K", net.K, "V", net.V, "attn", net.attn is not None)
print("concept names:", [c.name for c in net.concepts][:50])
print("n concepts:", len(net.concepts))

# does a raw concepts array exist in the npz?
for k in z.files:
    if "concept" in k.lower():
        print("concept-ish key:", k, z[k].shape, str(z[k])[:300])
