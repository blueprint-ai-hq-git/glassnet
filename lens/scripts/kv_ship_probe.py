#!/usr/bin/env python3
"""kv_ship_probe.py - which ship can the KV cache hook into, and what does it look like?

Prints only facts read off the checkpoint (no claims). Used before wiring the cache into decode.py.
"""
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "lens")
sys.path.insert(0, "lens/scripts")

import lens  # noqa: E402

for ship in sys.argv[1:] or ["lens/runs/chat/cl_d256_attn_12_ep12.npz",
                             "lens/runs/chat/sb_B1_d512_attn_ep96.npz"]:
    if not os.path.exists(ship):
        print(f"{ship}: MISSING")
        continue
    t0 = time.time()
    net = lens.load(ship, device="cpu")
    n = sum(p.data.size for p in net.parameters())
    print(f"{os.path.basename(ship)}: V={net.V} D={net.D} W={net.W} K={net.K} "
          f"attn={net.attn is not None} numeric={getattr(net, 'numeric', None)} mtp={net.mtp} "
          f"params={n:,} load={time.time()-t0:.1f}s")
