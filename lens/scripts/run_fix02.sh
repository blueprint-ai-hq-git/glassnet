#!/bin/bash
# run_fix02.sh — FIX-02: graded concepts instead of on/off.
#
# The two-phase experiment showed perfect concepts (BCE 0.00168) next to a near-random language
# model (CE 5.0158). The answer layer cannot predict 1,500 words from nine on/off values, so the
# bottleneck is the SHARPNESS of the concepts -- not the strength of their supervision, which is
# what three earlier attempts all attacked and all failed to fix.
#
# ONE CHANGE: the concept targets become graded (the fraction of the window the concept occupies)
# and the loss becomes a weighted squared error, instead of the balanced BCE that drives every
# neuron to a hard 0 or 1. lam_concept stays at 0.3, the cheapest setting that reached 9/9.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export BLOCKER_DEVICE=gpu
export BLOCKER_ONLY=readable
export BLOCKER_EPOCHS="${F2_EPOCHS:-60}"
export BLOCKER_SOFT=1
export BLOCKER_LC="${F2_LC:-0.3}"
export BLOCKER_OUT="fix02_soft_lc${F2_LC:-0.3}"

echo "=== recording the two-phase failure ==="
.venv-gpu/bin/python3 -u lens/scripts/record_twophase.py 2>&1 | tail -2

echo
echo "=== gradient check for the NEW concept_mse op (finite differences) ==="
.venv-gpu/bin/python3 - <<'PY'
import sys
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet")
import numpy as np
import lens
from lens import backend as B
from lens.concepts import Concept

V, W = 40, 6
rng = np.random.default_rng(0)
X = rng.integers(0, 20, size=(64, W)).astype(np.int64)
X[::2, 0] = 30
y = (X[:, 0] % V).astype(np.int64)
concepts = [Concept("c30", [30], support=0.5), Concept("c7", [7], support=0.3)]

net = lens.Net(vocab=V, concepts=concepts, d=16, window=W, seed=0,
               lam_concept=0.3, lam_act=0.0, device="cpu")

def total_loss():
    l, _ = net.loss(X, y, soft_concepts=True)
    return float(B.to_cpu(l.data))

for p in net.parameters():
    p.grad = None
l, parts = net.loss(X, y, soft_concepts=True)
l.backward()
print(f"  parts: {sorted(parts.keys())}   concept_mse={parts.get('concept_mse')}")

worst = 0.0
for p in net.parameters():
    if p.grad is None:
        print(f"  {p.name}: NO GRADIENT")
        continue
    g = B.to_cpu(p.grad)
    idx = [tuple(rng.integers(0, s) for s in p.data.shape) for _ in range(6)]
    eps = 1e-3
    for ix in idx:
        orig = float(B.to_cpu(p.data)[ix])
        a = B.to_cpu(p.data); a[ix] = orig + eps; p.data = B.to_dev(a)
        lp = total_loss()
        a = B.to_cpu(p.data); a[ix] = orig - eps; p.data = B.to_dev(a)
        lm = total_loss()
        a = B.to_cpu(p.data); a[ix] = orig; p.data = B.to_dev(a)
        num = (lp - lm) / (2 * eps)
        ana = float(g[ix])
        d = abs(num - ana) / (abs(num) + abs(ana) + 1e-9)
        worst = max(worst, d)
print(f"  worst relative difference (analytic vs finite differences): {worst:.3e}")
print(f"  {'PASS' if worst < 1e-3 else 'FAIL'} — gradient check for concept_mse")
PY

echo
echo "=== FIX-02: graded concepts, lc=$BLOCKER_LC, 60 epochs ==="
bash lens/scripts/run_blocker.sh 2>&1 | tail -30

echo
echo "=== comparison ==="
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs"
n = json.load(open(f"{P}/blocker/normal_summary.json"))["val_ppl"]
print(f"  matched normal ppl {n:.3f}")
print(f"  {'arm':<24}{'concepts':>10}{'ppl':>10}{'gap':>10}{'acc':>9}")
for tag, d in [("lc=0.1 (binary)", "blocker"), ("lc=0.3 (binary)", "fix01_lc0.3"),
               ("two-phase", "twophase"), ("FIX-02 soft lc=0.3", "fix02_soft_lc0.3")]:
    f = f"{P}/{d}/readable_summary.json"
    if not os.path.exists(f):
        f = f"{P}/{d}/summary.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    ppl = r["val_ppl"]
    passed, total = r["n_pass"], r["n_concepts"]
    print(f"  {tag:<24}{f'{passed}/{total}':>10}{ppl:>10.3f}{ppl-n:>+10.3f}"
          f"{r['val_acc']*100:>8.2f}%")
PY
