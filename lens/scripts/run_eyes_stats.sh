#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
.venv-gpu/bin/python3 -u - <<'PY' 2>&1 | tail -40
import os, sys
import numpy as np
PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ); sys.path.insert(0, PROJ + "/lens/scripts")
import eyes56 as E

for S in (28, 56):
    X, y = E.gen_dataset(S, 400, seed=3)
    A = X.reshape(-1, S, S)
    ink = (A > 0.5)
    frac = ink.mean(axis=(1, 2))
    dens = (np.abs(np.diff(A, axis=1)).sum(axis=(1, 2))
            + np.abs(np.diff(A, axis=2)).sum(axis=(1, 2))) / np.maximum(ink.sum(axis=(1, 2)), 1)
    q = S // 4
    corner = A[:, :q, :q].mean(axis=(1, 2))
    inner = A[:, q:3*q, q:3*q].mean(axis=(1, 2))
    print(f"  S={S}:  class  mean_dens  frac  corner_mean  inner_mean")
    for ci, name in enumerate(E.NAMES):
        m = y == ci
        print(f"      {name:<11} {dens[m].mean():6.2f}  {frac[m].mean():.3f}  "
              f"{corner[m].mean():.4f}   {inner[m].mean():.3f}")
    print(f"      dens*S: ring {(dens*S)[y==E.NAMES.index('ring')].mean():.0f} "
          f"thincross {(dens*S)[y==E.NAMES.index('thincross')].mean():.0f} "
          f"thick {(dens*S)[y==E.NAMES.index('thickcross')].mean():.0f} "
          f"circle {(dens*S)[y==0].mean():.0f}")
PY
