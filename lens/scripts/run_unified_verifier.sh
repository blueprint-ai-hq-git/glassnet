#!/bin/bash
# PLAN.txt line 370: verifier mode does not converge (9.0% trained / 0.0% unseen). Diagnose the mechanism and
# test the fix. PREDICTIONS P51a-i, written before any code was changed: lens/runs/UNIFIED-VERIFIER-PREDICTIONS.txt
# Output: lens/runs/UNIFIED-VERIFIER.txt (appended, never overwritten) + the byte-diff check for the default path.
# Short (a couple of minutes), CPU only, so it runs in the FOREGROUND: the GPU training arm is running and this
# job must not sit in the queue behind it.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
{
  echo "=== VERIFIER MODE: MECHANISM AND FIX $(date '+%F %H:%M') ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/unified_verifier_test.py
  echo "exit=$? (unified_verifier_test.py)"

  # P51i/P55 - the smoke test must reproduce the STORED output byte for byte. After the verifier2 change
  # (2026-10-21 18:3x) the stored file lens/runs/UNIFIED-RUN.txt IS the new probe3 default, and the
  # pre-change output is kept as lens/runs/UNIFIED-RUN-pre-probe3.txt; this check now proves the smoke test
  # is deterministic on the new default rather than that the default did not move (it moved on purpose).
  echo
  echo "--- P51i/P55: the smoke test, byte-compared against the stored output ---"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u -m lens.unified > lens/runs/_tmp/unified_after_edit.txt 2>&1
  if diff -q lens/runs/UNIFIED-RUN.txt lens/runs/_tmp/unified_after_edit.txt >/dev/null; then
    echo "P51i/P55 PASS: python3 -m lens.unified is byte-identical to lens/runs/UNIFIED-RUN.txt ($(wc -l < lens/runs/UNIFIED-RUN.txt) lines)"
  else
    echo "P51i/P55 FAIL: the stored output moved. diff:"
    diff lens/runs/UNIFIED-RUN.txt lens/runs/_tmp/unified_after_edit.txt | head -30
  fi

  # P55e - the change is inside the METHOD, not the shipped trainer. This project is NOT a git repo
  # (measured: `git rev-parse` says "not a git repository"), so the check is the recorded content hash and
  # mtime of the file, taken at 18:34 on 2026-10-21 before this turn's edits and after the 15:35 edit that
  # the in-flight comparison already runs.
  echo
  echo "--- P55e: the shipped trainer must be untouched by this turn ---"
  P55E_MD5="64929ca223456f74e78a6c16ae8f5f9d"
  got=$(md5sum lens/scripts/train_chat.py | cut -d' ' -f1)
  mt=$(stat -c '%y' lens/scripts/train_chat.py)
  if [ "$got" = "$P55E_MD5" ]; then
    echo "P55e PASS: lens/scripts/train_chat.py md5 $got matches the recorded $P55E_MD5, mtime $mt"
  else
    echo "P55e FAIL: lens/scripts/train_chat.py md5 is $got, recorded $P55E_MD5, mtime $mt"
  fi
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/UNIFIED-VERIFIER.txt 2>&1
