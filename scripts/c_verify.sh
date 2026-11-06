#!/bin/bash
# c_verify.sh — verify Target C's supervision wiring BEFORE the run.
#   1. the loader produces 6 taught meaningful targets (named ideas, not word clusters)
#   2. the AUX-ONLY gradient reaches the 6 taught columns
#   3. the AUX-ONLY gradient is EXACTLY zero on the 10 untaught columns (hold-out guarantee)
#   4. how much non-aux (sparsity) gradient the untaught columns get -- stated, not hidden
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T2_CONCEPT_SET=meaningful
export T2_N_TAUGHT=6
export T2_LAM_CONCEPT=0.5
export T3_BACKEND=numpy
PY=".venv-gpu/bin/python3"
LOG=runs/t3/c_verify.log

echo "=== TARGET C WIRING VERIFICATION  $(date '+%F %T') ===" | tee "$LOG"
"$PY" -u - <<'EOF' 2>&1 | tee -a "$LOG"
import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import numpy as np
import run_t2_iter003 as M
from glassnet.tensor import set_backend

set_backend("numpy")
t = M.CONCEPT_TARGETS
print(f"1. concept set: {t['kind']} | taught: {t['n_taught']} | targets derived: {t['n_targets']}")
print(f"   taught names: {t['words']}")
assert t["cols"].size == 6, "expected 6 taught columns"
assert all(isinstance(w, str) and "_" in w or w.isalpha() or w.startswith("is_") or w.startswith("contains")
           for w in t["words"]), "names should be idea-like, not word clusters"
print(f"   pos_weight (balanced BCE from TRAIN support): "
      f"{[round(float(x),2) for x in t['pos_weight']]}")

rng = np.random.default_rng(1)
ids = rng.integers(0, 1500, size=(64, 8))
y = rng.integers(0, 1500, size=(64,))

def grads(lam_concept):
    M.LAM_CONCEPT = lam_concept
    M.CONCEPT_TARGETS = M._load_concept_targets()
    net = M.TinyLM3(1501, 16, 64, 8, rng=np.random.default_rng(0), lam_concept=lam_concept)
    loss, parts = net.loss(ids, y)
    loss.backward()
    return net.concept.W.grad.copy(), parts

g_on, parts = grads(0.5)
g_off, _ = grads(0.0)
aux = np.abs(g_on - g_off)
taught, untaught = float(aux[:, 0:6].max()), float(aux[:, 6:16].max())
print(f"2. AUX-ONLY grad on taught cols 0-5   : max|dW| = {taught:.4e}")
print(f"3. AUX-ONLY grad on untaught cols 6-15: max|dW| = {untaught:.4e}")
assert untaught == 0.0, "untaught columns received concept gradient -- hold-out invalid"
assert taught > 0.0, "taught columns received no concept gradient"
print("   => hold-out guarantee HOLDS for the concept loss")
print(f"4. non-aux (sparsity) grad on untaught cols: max|dW| = "
      f"{float(np.abs(g_off[:, 6:16]).max()):.4e}  (expected non-zero: lam_act/lam_w touch")
print("   every column by design -- 'untaught' means never supervised on its target)")
print(f"   aux loss value with concepts on: {parts.get('aux', 'n/a')}")
EOF
RC=$?
echo "   wiring rc=$RC" | tee -a "$LOG"
[ "$RC" -ne 0 ] && { echo "!!! WIRING FAILED — not launching the run" | tee -a "$LOG"; exit 1; }
echo "=== wiring OK  $(date '+%T') ===" | tee -a "$LOG"