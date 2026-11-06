#!/bin/bash
# End-to-end proof that the clobber guard actually fires inside the real trainer.
#
# The unit test (tests/test_ckpt_guard.py) proves the function; this proves the CALL SITE, which is
# the part that was broken: train_chat.py must move an existing <out>_best.npz aside before a fresh
# run writes its own. Nothing is trained to convergence here - two 1-epoch runs on 1,500 windows, CPU.
#
# Prediction, written BEFORE running:
#   P1. Run 1 (fresh tag guard_e2e) prints no CLOBBER GUARD line (nothing existed) and writes
#       lens/runs/chat/guard_e2e_best.npz. Expect its held-out acc > 0%.
#   P2. Run 2 (same tag, fresh, different seed) prints exactly one CLOBBER GUARD line naming a sha256,
#       and lens/runs/chat/archive/best_clobber_guard_*/guard_e2e_best.npz.<sha8> exists with that hash.
#   P3. Run 1's checkpoint survives byte-for-byte in the archive (sha256 recorded before/after match).
#   P4. Run 2's own best.npz is a DIFFERENT file from run 1's (different weights, different hash).
#   If P2 fails, the guard is not wired and the patch must be reverted from
#   lens/runs/archive/train_chat_prepatch_*/train_chat.py, not left half-done.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
LOG="lens/runs/BEST-CKPT-GUARD-E2E.txt"
PY=".venv-gpu/bin/python3"
CK="lens/runs/chat"
TAG="guard_e2e"

{
  echo "=== $(date '+%F %T') clobber-guard end-to-end"

  if pgrep -af "train_chat.py" | grep -v run_guard_e2e >/dev/null; then
    echo "BLOCKED: another train_chat.py is alive; not competing with it."
    pgrep -af "train_chat.py" | grep -v run_guard_e2e
    exit 3
  fi
  if ! nice -19 ionice -c3 "$PY" lens/scripts/apply_best_ckpt_guard.py --check; then
    echo "BLOCKED: the guard is not applied to train_chat.py - run run_apply_best_ckpt_guard.sh first."
    exit 4
  fi

  # Start from a clean tag so P1's \"nothing existed\" is true. Only this run's own leftovers are moved.
  if [ -e "$CK/${TAG}_best.npz" ] || ls "$CK/archive"/best_clobber_guard_*/"${TAG}_best.npz".* >/dev/null 2>&1; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p "$CK/archive/guard_e2e_rerun_$STAMP"
    for f in "$CK/${TAG}"_*; do [ -e "$f" ] && mv "$f" "$CK/archive/guard_e2e_rerun_$STAMP/"; done
    echo "moved an earlier guard_e2e run aside to $CK/archive/guard_e2e_rerun_$STAMP (not deleted)"
  fi

  run() {  # $1 = seed, $2 = logfile
    nice -19 ionice -c3 "$PY" -u lens/scripts/train_chat.py \
      --d 64 --window 32 --epochs 1 --batch 128 --seed "$1" --device cpu \
      --max-windows 1500 --out "$TAG" 2>&1 | tee "$2"
  }

  echo "--- run 1 (seed 0)"
  run 0 "lens/runs/GUARD-E2E-run1.txt" | tail -n 4
  echo "best.npz after run 1: $(sha256sum "$CK/${TAG}_best.npz" 2>/dev/null | cut -c1-16)"

  echo "--- run 2 (seed 1, same tag, fresh)"
  run 1 "lens/runs/GUARD-E2E-run2.txt" >/dev/null
  GUARD_LINE=$(grep -c "CLOBBER GUARD" "lens/runs/GUARD-E2E-run2.txt")
  echo "CLOBBER GUARD lines in run 2: $GUARD_LINE"
  grep "CLOBBER GUARD" "lens/runs/GUARD-E2E-run2.txt" || true

  NEW_SHA=$(sha256sum "$CK/${TAG}_best.npz" | cut -d' ' -f1)
  ARCH=$(ls "$CK/archive"/best_clobber_guard_*/"${TAG}_best.npz".* 2>/dev/null | grep -v '\.sha256$' | tail -1)
  echo "run 2 best.npz sha256: $NEW_SHA"
  echo "archived file: ${ARCH:-NONE}"
  if [ -n "${ARCH:-}" ]; then
    OLD_SHA=$(sha256sum "$ARCH" | cut -d' ' -f1)
    echo "archived sha256: $OLD_SHA   receipt: $(cat "$ARCH.sha256" 2>/dev/null)"
    [ "$OLD_SHA" != "$NEW_SHA" ] && echo "P4 ok: run 2 wrote different weights" || echo "P4 FAIL: same file"
  else
    echo "P2 FAIL: nothing was archived - the guard did not fire at the call site"
  fi

  echo "--- verdict"
  if [ "${GUARD_LINE:-0}" -ge 1 ] && [ -n "${ARCH:-}" ]; then
    echo "PASS: the guard fired once and run 1's checkpoint survives in $(dirname "$ARCH")"
  else
    echo "FAIL: see P2 above. Revert from lens/runs/archive/train_chat_prepatch_*/train_chat.py"
  fi
  echo "DONE $(date '+%F %T')"
} 2>&1 | tee -a "$LOG"
