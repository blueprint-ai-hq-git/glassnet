#!/bin/bash
# CUSTOM ARCHITECTURE: MAKE THE NAMED CONCEPTS LOAD-BEARING (2026-11-20 12:35)
#
# Flags here were VERIFIED against `train_chat.py --help` before writing this file. My first draft used `--direct 1/0` and
# `--plain`, neither of which exists: --direct is a switch (presence = on) and there is no plain architecture control in this
# trainer (the plain matched control lives in train_image.py, where it already measured -0.08 = parity). A script with invented
# flags would have failed 3 minutes in and wasted the wake.
#
# WHAT IS ALREADY MEASURED:
#   * readable IMAGE model vs matched plain control: **-0.08 points** = parity. "Readable costs accuracy" is FALSE at matched params.
#   * chat model: the direct pathway carries 82-91% of the decision and explain() cannot see it. The concept layer is a passenger.
#     Concept "share" was reported with a ratio that could overstate it up to 60x; that metric is now fixed (concept_signed /
#     reconstruct / cancellation), so both arms are measured with the corrected instrument.
#
# CLOBBER BUG I COMMITTED ON THE FIRST RUN (2026-11-20 12:55): this script passed no --out, so both arms wrote the default
# `run1` and the second arm OVERWROTE the first arm's checkpoint. The accuracy comparison survived because the numbers are in the
# log, but the offline concept-share measurement can no longer be done on the first arm's weights. Fixed: --out "arch_$1_m${MAXW}w" per arm.
#
# THE ONE CHANGE: `--direct` (position-aware path from the token window to the answer) present in arm A, absent in arm B.
# Arm B's answer has to flow through named concepts -> sparse rules. That is the difference between "readable because we also
# train concepts" and "readable because it cannot answer any other way". Training method, data, width, window, seed and concept
# supervision are identical in both.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 arm B's named layer carries a median share ABOVE arm A's, measured with the corrected metric (not the old ratio).
#   P2 arm B lands within 2.0 points of arm A on held-out accuracy. Forcing structure for more than that is a bad trade.
#   P3 arm B's concept audit stays at or above the bar (AUROC >= 0.55, fire floor >= 0.02) - concepts still mean what they say
#      once they are load-bearing.
# FALSIFIER: if arm B's concept share is not measurably above arm A's, forcing the answer through the named layer does NOT make
#   the model more readable - the concepts are decorative even as a bottleneck - and this architecture line CLOSES rather than
#   being refined again.
#
# DO NOT LAUNCH WHILE HE IS GAMING. Both arms read the 1.1 GB chat corpus from /dev/sdc5, the damaged 89%-full NTFS volume that
# is also a suspect in his freezes. That IO pattern during a game is exactly what to avoid. Launch when idle AND when the
# memory pair (glassnet-off2pair) has finished, so the cores are not shared three ways.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
# BUDGET: the trainer's own note says the FULL corpus under --method ours costs 10.8-13.8 h per arm, which is far past any
# wake. MAXW caps the corpus so one arm is a known, bounded run; EPOCHS likewise. Both are env-overridable so this can be
# smoke-tested cheaply before anyone commits hours to it.
MAXW=${MAXW:-20000}
EPOCHS=${EPOCHS:-1}
OUT=${OUT:-lens/runs/CUSTOM-ARCH-DIRECT.txt}
: > "$OUT"
echo "=== custom architecture: concepts load-bearing vs bypassed, $(date '+%F %T') ===" | tee -a "$OUT"

run_arm() {  # $1 = label, $2 = "" or "--direct"
  echo "--- $1 ---" | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME nice -n 19 ionice -c3 taskset -c 6,7 \
    ${PYTHON:-python3} -u lens/scripts/train_chat.py \
    --d 128 --window 8 --epochs "$EPOCHS" --batch 256 --lr 0.4 --lam-concept 0.3 \
    --concept-target window --method ours --seed 0 --max-windows "$MAXW" --out "arch_$1_m${MAXW}w" $2 \
    >> "lens/runs/chain_arch_$1.out" 2>&1
  local rc=$?
  echo "  rc=$rc" | tee -a "$OUT"
  if [ "$rc" -ne 0 ]; then echo "  ARM $1 FAILED (rc=$rc) - stopping" | tee -a "$OUT"; return 1; fi
  grep -iE 'held-out|concept_share|concept share|direct_share|cancell|auroc|fire' "lens/runs/chain_arch_$1.out" \
    | tail -8 | tee -a "$OUT"
  return 0
}

run_arm direct_on  "--direct" || exit 1
run_arm direct_off ""        || exit 1

echo "--- P1/P2/P3 verdict ---" | tee -a "$OUT"
echo "  compare the concept-share lines above (P1), the held-out accuracies (P2), the audit lines (P3)" | tee -a "$OUT"
echo "CUSTOM-ARCH COMPLETE" | tee -a "$OUT"
