#!/bin/bash
# a2_verify.sh — verify the A2 supervision wiring BEFORE the long run.
#
# IMPORTANT SUBTLETY: the held-out columns still receive gradient from the ARCHITECTURE-WIDE
# sparsity penalties (lam_act, lam_w), which touch every column by design. So a total
# |dW| == 0 assertion is wrong. What must be exactly zero is the CONCEPT AUX contribution
# specifically -- that is what "untaught" means here. It is isolated by differencing two
# identical models that differ ONLY in lam_concept.
#
# Reports:
#   1. the target loader produces 24 taught targets (not 32)
#   2. the aux-only gradient reaches the 24 taught columns
#   3. the aux-only gradient is EXACTLY zero on the 8 held-out columns
#   4. how much NON-aux (sparsity) gradient the held-out columns receive, stated plainly
#   5. the standard aux gradient check still passes with this concept set
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T2_CONCEPT_SET=clusters32
export T2_N_TAUGHT=24
export T2_LAM_CONCEPT=0.5
export T3_BACKEND=numpy
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/a2_verify.log

echo "=== A2 WIRING VERIFICATION  $(date '+%F %T') ===" | tee "$LOG"

"$PY" -u - <<'EOF' 2>&1 | tee -a "$LOG"
import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import numpy as np
import run_t2_iter003 as M
from glassnet.tensor import set_backend

set_backend("numpy")
t = M.CONCEPT_TARGETS
print(f"1. concept set: {t['kind']} | taught columns: {t['n_taught']} "
      f"| pre-registered targets: {t['n_targets']}")
print(f"   cols taught: {t['cols'].min()}..{t['cols'].max()} (count {t['cols'].size}); "
      f"first '{t['words'][0]}' last taught '{t['words'][-1]}'")
assert t["cols"].size == 24, "expected 24 taught columns"

rng = np.random.default_rng(1)
ids = rng.integers(0, 1500, size=(64, 8))
y = rng.integers(0, 1500, size=(64,))

def grads(lam_concept):
    # Set the module globals directly (no reload) and pass lam_concept EXPLICITLY:
    # the constructor's default is bound at definition time, so a reload would not
    # change it and the two models would otherwise be identical.
    M.LAM_CONCEPT = lam_concept
    M.CONCEPT_TARGETS = M._load_concept_targets()
    net = M.TinyLM3(1501, 32, 64, 8, rng=np.random.default_rng(0),
                    lam_concept=lam_concept)
    loss, parts = net.loss(ids, y)
    loss.backward()
    return net.concept.W.grad.copy(), parts

g_on, parts_on = grads(0.5)
g_off, _ = grads(0.0)
aux_only = np.abs(g_on - g_off)               # everything except the aux term cancels

taught = float(aux_only[:, 0:24].max())
held = float(aux_only[:, 24:32].max())
print(f"2. AUX-ONLY gradient on taught cols 0-23   : max|dW| = {taught:.4e}")
print(f"3. AUX-ONLY gradient on held-out cols 24-31: max|dW| = {held:.4e}")
print(f"   aux reaches held-out columns at all: {held > 0.0}")
assert held == 0.0, "held-out columns received CONCEPT gradient -- hold-out invalid"
assert taught > 0.0, "taught columns received no concept gradient"
print("   => hold-out guarantee HOLDS for the concept loss")

sp = float(np.abs(g_off[:, 24:32]).max())
print(f"4. NON-aux (sparsity) gradient on held-out cols: max|dW| = {sp:.4e}")
print("   NOTE, stated plainly: the held-out columns are NOT pristine-random. The")
print("   architecture-wide sparsity penalties shape them. 'Untaught' here means")
print("   'never supervised on their pre-registered target', which is what the")
print("   hold-out check measures. Recorded so it cannot be misread later.")
EOF
RC=$?
echo "   wiring rc=$RC" | tee -a "$LOG"

echo "=== 5. standard aux gradient check with this concept set ===" | tee -a "$LOG"
T2_LAM_CONCEPT=0.5 bash "/run/media/yahya/HARD DISK 4/GlassNet/scripts/aux_gradcheck.sh" 2>&1 \
  | grep -E "RESULT|FAIL" | tail -3 | tee -a "$LOG"
echo "=== done  $(date '+%T') ===" | tee -a "$LOG"