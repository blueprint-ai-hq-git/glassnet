#!/bin/bash
# d_verify.sh — verify TARGET D's construction BEFORE the run.
#   The whole point of D is that there are NO untaught columns, so this checks exactly that:
#     * the loader gives 6 taught targets, all with idea-like names
#     * K = 6, so every column the answer layer can see is one of them
#     * the aux gradient reaches ALL 6 columns (none is starved)
#   If K were 16, ten columns would be unnamed and coverage could not be 100%.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T2_CONCEPT_SET=meaningful
export T2_N_TAUGHT=6
export T2_LAM_CONCEPT=0.5
export T3_K=6
export T3_BACKEND=numpy
PY=".venv-gpu/bin/python3"
LOG=runs/t3/d_verify.log

echo "=== TARGET D WIRING VERIFICATION  $(date '+%F %T') ===" | tee "$LOG"
"$PY" -u - <<'EOF' 2>&1 | tee -a "$LOG"
import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import numpy as np
import run_t2_iter003 as M
from glassnet.tensor import set_backend

set_backend("numpy")
K = int(os.environ["T3_K"])
t = M.CONCEPT_TARGETS
print(f"1. K = {K} | taught = {t['n_taught']} | targets derived = {t['n_targets']}")
print(f"   names: {t['words']}")
assert t["n_taught"] == K, (f"K={K} but {t['n_taught']} taught -> {K - t['n_taught']} UNNAMED "
                            "columns would exist and coverage could not be 100%")
print("   => every column the answer layer can see is a named concept (0 unnamed)")

rng = np.random.default_rng(1)
ids = rng.integers(0, 1500, size=(64, 8))
y = rng.integers(0, 1500, size=(64,))

def grads(lam):
    M.LAM_CONCEPT = lam
    M.CONCEPT_TARGETS = M._load_concept_targets()
    net = M.TinyLM3(1501, K, 64, 8, rng=np.random.default_rng(0), lam_concept=lam)
    loss, _ = net.loss(ids, y)
    loss.backward()
    return net.concept.W.grad.copy()

g_on, g_off = grads(0.5), grads(0.0)
aux = np.abs(g_on - g_off)
print(f"2. AUX-ONLY grad, per column (all {K} must be non-zero):")
per = [float(aux[:, c].max()) for c in range(K)]
for c, v in enumerate(per):
    print(f"     col {c} {t['words'][c]:<22} max|dW| = {v:.4e}")
assert all(v > 0 for v in per), "a column received NO concept gradient -> it would be unnamed"
print(f"3. starved columns: 0 of {K}")
print(f"4. params: {sum(p.data.size for p in M.TinyLM3(1501, K, 64, 8, rng=np.random.default_rng(0)).parameters())}")
EOF
RC=$?
echo "   wiring rc=$RC" | tee -a "$LOG"
[ "$RC" -ne 0 ] && { echo "!!! WIRING FAILED — not launching" | tee -a "$LOG"; exit 1; }
echo "=== wiring OK  $(date '+%T') ===" | tee -a "$LOG"