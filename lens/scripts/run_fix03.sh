#!/bin/bash
# run_fix03.sh — FIX-03: graded concepts with the LOG loss (soft labels).
#
# FIX-02 used squared error on a graded target. It helped the language (ppl 99.313, acc 30.03% --
# both the best of any arm) and killed every concept (0/9), because squared error on a RARE target
# optimises to the MEAN: every neuron sat near zero and never fired. The two facts are the same
# fact -- a constant concept layer is hardly a bottleneck.
#
# FIX-03 keeps the graded target and changes only the loss to the BCE, which cannot collapse to the
# mean because its gradient stays large while the neuron is wrong.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export BLOCKER_DEVICE=gpu
export BLOCKER_ONLY=readable
export BLOCKER_EPOCHS="${F3_EPOCHS:-60}"
export BLOCKER_SOFT=bce
export BLOCKER_LC="${F3_LC:-0.3}"
export BLOCKER_OUT="fix03_softbce_lc${F3_LC:-0.3}"

echo "=== recording FIX-02 ==="
.venv-gpu/bin/python3 -u lens/scripts/record_fix02.py 2>&1 | tail -2

echo
echo "=== gradient check: soft-BCE path, finite differences on concept.W ==="
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
X[::3, 1] = 31
y = (X[:, 0] % V).astype(np.int64)
concepts = [Concept("c30", [30], support=0.5), Concept("c31", [31], support=0.33)]

net = lens.Net(vocab=V, concepts=concepts, d=16, window=W, seed=0,
               lam_concept=0.3, lam_act=0.0, device="cpu")

def total_loss():
    l, _ = net.loss(X, y, soft_concepts="bce")
    return float(B.to_cpu(l.data))

for p in net.parameters():
    p.grad = None
l, parts = net.loss(X, y, soft_concepts="bce")
l.backward()
print(f"  parts: {sorted(parts.keys())}   concept_aux={parts.get('concept_aux')}")

worst = 0.0
for p in net.parameters():
    if p.grad is None:
        print(f"  {p.name}: NO GRADIENT")
        continue
    g = B.to_cpu(p.grad)
    for _ in range(6):
        ix = tuple(int(rng.integers(0, s)) for s in p.data.shape)
        a = B.to_cpu(p.data); orig = float(a[ix])
        eps = 1e-3
        a[ix] = orig + eps; p.data = B.to_dev(a); lp = total_loss()
        a = B.to_cpu(p.data); a[ix] = orig - eps; p.data = B.to_dev(a); lm = total_loss()
        a = B.to_cpu(p.data); a[ix] = orig; p.data = B.to_dev(a)
        num = (lp - lm) / (2 * eps)
        ana = float(g[ix])
        worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-9))
print(f"  worst relative difference (analytic vs finite differences): {worst:.3e}")
print(f"  {'PASS' if worst < 1e-3 else 'FAIL'} — gradient check for the soft-BCE path")
PY

echo
echo "=== FIX-03: graded targets + LOG loss, lc=$BLOCKER_LC, 60 epochs ==="
bash lens/scripts/run_blocker.sh 2>&1 | tail -28

echo
echo "=== every arm so far ==="
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs"
n = json.load(open(f"{P}/blocker/normal_summary.json"))["val_ppl"]
print(f"  matched normal ppl {n:.3f}   (bar ~6)")
print(f"  {'arm':<26}{'concepts':>10}{'ppl':>10}{'gap':>10}{'acc':>9}")
for tag, d in [("lc=0.1 binary", "blocker"), ("lc=0.3 binary", "fix01_lc0.3"),
               ("lc=0.5 binary", "fix01_lc0.5"), ("lc=1.0 binary", "fix01_lc1"),
               ("two-phase", "twophase"), ("FIX-02 soft MSE", "fix02_soft_lc0.3"),
               ("FIX-03 soft BCE", "fix03_softbce_lc0.3")]:
    f = f"{P}/{d}/readable_summary.json"
    if not os.path.exists(f):
        f = f"{P}/{d}/summary.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    ppl = r["val_ppl"]
    passed, total = r["n_pass"], r["n_concepts"]
    print(f"  {tag:<26}{f'{passed}/{total}':>10}{ppl:>10.3f}{ppl-n:>+10.3f}"
          f"{r['val_acc']*100:>8.2f}%")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md FIX-01.md /home/yahya/ 2>/dev/null
echo "done"
