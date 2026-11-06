#!/bin/bash
# WHAT IS IN the image/medical caches (keys + shapes + split), so the leak audit can be written
# against the real arrays instead of assumptions.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import numpy as np, glob, os
for p in ("data/fashion_mnist_cache.npz", "data/loop_pathmnist_gray.npz", "data/loop_split.npz"):
    if not os.path.exists(p):
        print(p, "MISSING"); continue
    z = np.load(p, allow_pickle=True)
    print(f"--- {p}")
    for k in z.files[:10]:
        a = z[k]
        try:
            print(f"    {k:<16} {str(a.shape):<18} {a.dtype}  min {a.min()} max {a.max()}")
        except Exception:
            print(f"    {k:<16} {str(a.shape):<18} {a.dtype}")
PY