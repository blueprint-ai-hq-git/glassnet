"""mem_namer.py — measure the namer's peak memory on the full TRAIN set.

The re-audit was OOM-killed at ~4.18 GB RSS inside the Hermes worker cgroup.
The namer calls net.forward(Xtr) with all 182,377 windows at once, so this measures
that spike directly instead of assuming it.
"""
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))
os.environ.setdefault("T3A_BACKEND", "numpy")
os.environ.setdefault("T3A_K", "32")

import numpy as np
from glassnet.tensor import set_backend
import t3_audit as A
import run_t2_iter003 as M

set_backend("numpy")
MODE = sys.argv[1] if len(sys.argv) > 1 else "full"


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024
    return -1.0


def rss_now():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return -1.0


d = A.tiny_data.build(win=A.WIN, vocab_cap=A.VOCAB_CAP)
V, K, D, WIN = d["V"], A.K, M.D, A.WIN
print(f"data: Xtr {d['Xtr'].shape}, Xva {d['Xva'].shape}, V={V}, K={K}, D={D}, W={WIN}")
print(f"RSS after data load: {rss_now():.1f} MB")

ck = os.path.join(PROJ, "runs", "t3", "ckpt", "readable_K32_lr0.5_lc10_seed0.npz")
net = M.TinyLM3(V, K, D, WIN, rng=np.random.default_rng(0))
z = np.load(ck)
for p in net.parameters():
    if p.name in z:
        p.data = np.array(z[p.name])
print(f"RSS after model load: {rss_now():.1f} MB")

X = np.asarray(d["Xtr"])
base = rss_now()
print(f"mode={MODE}  Xtr {X.shape}  V={V} K={K}  RSS after setup {base:.1f} MB")

# Each mode must run in its OWN process: Python's allocator keeps freed memory
# resident, so measuring both in one process makes the second number meaningless.
if MODE == "full":
    f = net.forward(X)
    h = np.asarray(f["concepts"].data)
    print(f"  full-set forward -> concepts {h.shape}")
    del f, h
elif MODE == "namer":
    # the REAL fixed path: the namer now uses _fired_matrix (chunked + gc)
    names = M.auto_name_t2c_pa(net, d, X, min_support=0.01)
    print(f"  namer produced {len(names)} names; first: {names[0][:40]!r}")
else:
    CH = 20000
    out = None
    import gc
    for i in range(0, len(X), CH):
        hh = np.asarray(net.forward(X[i:i + CH])["concepts"].data)
        if out is None:
            out = np.empty((len(X), hh.shape[1]), dtype=np.float32)
        out[i:i + CH] = hh
        if MODE == "chunked_gc":
            del hh
            n = gc.collect()
        print(f"    chunk {i//CH:3d} ({i:>6}..{i+CH:>6})  RSS {rss_now():8.1f} MB  "
              f"HWM {rss_mb():8.1f} MB" + (f"  gc freed {n}" if MODE == "chunked_gc" else ""))
    print(f"  chunked forward ({CH}/step) -> concepts {out.shape}")

peak = rss_mb()
print(f"  peak RSS (high-water) {peak:.1f} MB   extra over setup: "
      f"{peak - base:.1f} MB")
