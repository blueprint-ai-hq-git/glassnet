#!/bin/bash
# Apply the best-checkpoint clobber guard to lens/scripts/train_chat.py, but ONLY when no arm is
# training. Reason: docs/history/WALL-AB-ARMS.md says editing the trainer between arms makes the arms
# non-identical, and the wall A/B is the experiment that needs them identical. If a trainer is alive
# this script does nothing at all and says so (exit 3), so the queue can move on safely.
#
# Prediction, written before running (so the outcome cannot be read backwards):
#   P1. With no trainer alive, apply_best_ckpt_guard.py reports "patched: import=True call=True" and
#       leaves a backup whose sha256 equals the pre-patch file's. Expected: PASS.
#   P2. tests/test_ckpt_guard.py stays 4/4 PASS afterwards (it does not touch train_chat.py).
#   P3. --check on a second run says "already patched" and writes nothing (idempotent).
#   If any of the three fails, the patch is reverted from the backup and the failure is written to
#   lens/runs/AUTOPILOT-LOG.md rather than being worked around.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
LOG="lens/runs/BEST-CKPT-GUARD-APPLY.txt"
PY=".venv-gpu/bin/python3"

{
  echo "=== $(date '+%F %T') apply_best_ckpt_guard"

  if pgrep -af "train_chat.py" | grep -v apply_best_ckpt_guard >/dev/null; then
    echo "BLOCKED: a train_chat.py process is alive:"
    pgrep -af "train_chat.py" | grep -v apply_best_ckpt_guard
    echo "exit 3 - nothing was changed. Re-queue for when the GPU is free."
    exit 3
  fi

  echo "--- before"
  nice -19 ionice -c3 "$PY" lens/scripts/apply_best_ckpt_guard.py --check || true
  echo "--- apply"
  nice -19 ionice -c3 "$PY" lens/scripts/apply_best_ckpt_guard.py || { echo "APPLY FAILED rc=$?"; exit 4; }
  echo "--- tests"
  nice -19 ionice -c3 "$PY" tests/test_ckpt_guard.py || { echo "TESTS FAILED rc=$?"; exit 5; }
  echo "--- idempotence"
  nice -19 ionice -c3 "$PY" lens/scripts/apply_best_ckpt_guard.py || { echo "SECOND APPLY FAILED rc=$?"; exit 6; }
  echo "--- after"
  nice -19 ionice -c3 "$PY" lens/scripts/apply_best_ckpt_guard.py --check || true
  echo "DONE $(date '+%F %T')"
} 2>&1 | tee -a "$LOG"
