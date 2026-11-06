"""precheck_code_redesign.py — real-vocab pre-gate for the redesigned code concept set."""
import os
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, P)
sys.path.insert(0, P + "/lens/scripts")

import numpy as np
import code_data
import code_concepts

d = code_data.build()
concepts, spec = code_concepts.build(d["vocab"], d["ids"], d["split"] == 0,
                                     min_support=0.005, windows=None)
print("kept:", spec["n_kept"], "| dropped:", spec["n_dropped"], "| vocab:", len(d["vocab"]))
for dr in spec["dropped"][:8]:
    print("  dropped:", dr["name"], "-", dr["reason"])

X, _ = code_data.make_windows(d["ids"], d["bounds"], d["split"], 16, which=1)
X = X[:4000]
names = [c["name"] for c in concepts]
T = np.stack([np.isin(X, c["ids"]).any(axis=1) for c in concepts], axis=1).astype(np.float64)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    dd = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / dd) if dd > 1e-12 else 0.0


n = len(names)
M = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        M[i, j] = abs(corr(T[:, i], T[:, j]))
np.fill_diagonal(M, 0.0)
pairs = [(names[i], names[j], float(M[i, j])) for i in range(n) for j in range(i + 1, n)]
bad = [p for p in pairs if p[2] > 0.6]
risk = [p for p in pairs if 0.2 < p[2] <= 0.6]
bad.sort(key=lambda t: -t[2])
risk.sort(key=lambda t: -t[2])
print(f"REDESIGNED set @ v{len(d['vocab'])}: {n} concepts | BLOCK pairs >0.6: {len(bad)} "
      f"| risk 0.2-0.6: {len(risk)}")
for a, b, c in bad[:6]:
    print(f"   BLOCK {a} <-> {b} {c:.2f}")
for a, b, c in risk[:8]:
    print(f"   risk  {a} <-> {b} {c:.2f}")
print("PRE-GATE:", "BLOCKED (redesign again)" if bad else "CLEAN (no twin pairs; retrain with this set)")
sys.exit(2 if bad else 0)