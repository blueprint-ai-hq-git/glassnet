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

# SELF-LOCK (added 2026-11-06 18:0x): the chain wakes an agent the instant a short item ends, and two
# woken agents read the same queue line and run it twice. Measured today: three duplicate-agent
# collisions, one of which swept an in-flight edit into an unrelated commit and one of which clobbered a
# checkpoint. flock turns the second copy into a no-op instead of a race on train_chat.py.
exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy of $(basename "$0") holds the lock; nothing was changed" | tee -a "$LOG"
  exit 0
fi

# The trainer detector is shared, and it anchors the interpreter at the start of the command line.
# `pgrep -af "train_chat.py"` matched an AGENT's prompt on 2026-11-06 17:59 and blocked the next item
# for a trainer that did not exist: lens/scripts/lib_trainer_alive.sh has the measurement.
source lens/scripts/lib_trainer_alive.sh

{
  echo "=== $(date '+%F %T') apply_best_ckpt_guard"

  if trainer_alive; then
    echo "BLOCKED: a real train_chat.py run is alive:"
    trainer_alive_print
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
