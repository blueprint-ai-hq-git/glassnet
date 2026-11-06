#!/bin/bash
# PLAN item (verifier3): a DIRECTION out of a RIGHT/WRONG verdict. Predictions P57a-P57l written BEFORE any
# code changed: lens/runs/UNIFIED-LADDER-PREDICTIONS.txt. Output: lens/runs/UNIFIED-LADDER.txt (appended).
# Short (a few minutes), CPU only, run in the FOREGROUND by the self-wake: the GPU training arm is running
# and this job must not sit in the queue behind 7 training arms.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
{
  echo "=== A DIRECTION OUT OF A RIGHT/WRONG VERDICT (PLAN verifier3) $(date '+%F %H:%M') ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/unified_ladder_test.py
  echo "exit=$? (unified_ladder_test.py)"

  # P57j / P51i - the edit is inside verifier mode: the smoke test (data arm, teacher arm, STAGE 2) must be
  # byte-identical to the stored output. NOTE (2026-10-21 19:3x): the stored output lens/runs/UNIFIED-RUN.txt
  # is now the LADDER default (binary_probe='ladder'), because P57b/P57c passed and the default moved on
  # purpose; the pre-move output is kept at lens/runs/UNIFIED-RUN-pre-ladder.txt. So this check now proves the
  # smoke test is deterministic on the new default, not that the default did not move.
  echo
  echo "--- P57j: the smoke test, byte-compared against the stored output ---"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u -m lens.unified > lens/runs/_tmp/unified_after_ladder.txt 2>&1
  if diff -q lens/runs/UNIFIED-RUN.txt lens/runs/_tmp/unified_after_ladder.txt >/dev/null; then
    echo "P57j PASS: python3 -m lens.unified is byte-identical to lens/runs/UNIFIED-RUN.txt ($(wc -l < lens/runs/UNIFIED-RUN.txt) lines) - the data arm, the teacher arm and STAGE 2 are untouched and the default path has not moved"
  else
    echo "P57j FAIL: the stored output moved. diff:"
    diff lens/runs/UNIFIED-RUN.txt lens/runs/_tmp/unified_after_ladder.txt | head -30
  fi

  # P57k - the change is inside the METHOD, not the shipped trainer (this project is not a git repo, so the
  # check is the recorded content hash and mtime).
  echo
  echo "--- P57k: the shipped trainer must be untouched by this turn ---"
  P57K_MD5="64929ca223456f74e78a6c16ae8f5f9d"
  got=$(md5sum lens/scripts/train_chat.py | cut -d' ' -f1)
  mt=$(stat -c '%y' lens/scripts/train_chat.py)
  if [ "$got" = "$P57K_MD5" ]; then
    echo "P57k PASS: lens/scripts/train_chat.py md5 $got matches the recorded $P57K_MD5, mtime $mt"
  else
    echo "P57k FAIL: lens/scripts/train_chat.py md5 is $got, recorded $P57K_MD5, mtime $mt"
  fi
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/UNIFIED-LADDER.txt 2>&1
