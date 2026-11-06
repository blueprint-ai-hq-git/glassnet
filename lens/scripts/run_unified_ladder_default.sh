#!/bin/bash
# The follow-up to PLAN item (verifier3): P57b/P57c PASSED, so the DEFAULT MOVED to binary_probe='ladder'.
# Steps: keep the pre-move smoke output, re-record the smoke output on the new default, run the corrected
# checks (lens/scripts/unified_ladder_followup.py), re-check determinism, and re-record the trainer hash.
# Output: lens/runs/UNIFIED-LADDER.txt (appended, never overwritten).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
{
  echo
  echo "=== THE DEFAULT MOVES TO THE LADDER (PLAN verifier3) $(date '+%F %H:%M') ==="
  echo "--- keep the pre-move smoke output, then re-record on the new default ---"
  if [ ! -f lens/runs/UNIFIED-RUN-pre-ladder.txt ]; then
    cp lens/runs/UNIFIED-RUN.txt lens/runs/UNIFIED-RUN-pre-ladder.txt
    echo "kept: lens/runs/UNIFIED-RUN-pre-ladder.txt ($(wc -l < lens/runs/UNIFIED-RUN-pre-ladder.txt) lines, md5 $(md5sum lens/runs/UNIFIED-RUN-pre-ladder.txt | cut -d' ' -f1))"
  else
    echo "already kept: lens/runs/UNIFIED-RUN-pre-ladder.txt ($(wc -l < lens/runs/UNIFIED-RUN-pre-ladder.txt) lines)"
  fi
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u -m lens.unified > lens/runs/UNIFIED-RUN.txt 2>&1
  echo "re-recorded: lens/runs/UNIFIED-RUN.txt ($(wc -l < lens/runs/UNIFIED-RUN.txt) lines, md5 $(md5sum lens/runs/UNIFIED-RUN.txt | cut -d' ' -f1))"

  echo
  echo "--- the corrected checks and the arm-by-arm split of the move ---"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/unified_ladder_followup.py
  echo "exit=$? (unified_ladder_followup.py)"

  echo
  echo "--- determinism on the NEW default: the smoke test re-run must match the re-recorded output ---"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u -m lens.unified > lens/runs/_tmp/unified_after_ladder_default.txt 2>&1
  if diff -q lens/runs/UNIFIED-RUN.txt lens/runs/_tmp/unified_after_ladder_default.txt >/dev/null; then
    echo "P57l PASS: python3 -m lens.unified is byte-identical to the re-recorded lens/runs/UNIFIED-RUN.txt"
  else
    echo "P57l FAIL: the re-run differs from the stored output. diff:"
    diff lens/runs/UNIFIED-RUN.txt lens/runs/_tmp/unified_after_ladder_default.txt | head -20
  fi

  echo
  echo "--- the shipped trainer: this turn did not touch it, but a PARALLEL turn did ---"
  echo "lens/scripts/train_chat.py md5 $(md5sum lens/scripts/train_chat.py | cut -d' ' -f1) mtime $(stat -c '%y' lens/scripts/train_chat.py)"
  echo "lens/model.py           md5 $(md5sum lens/model.py | cut -d' ' -f1) mtime $(stat -c '%y' lens/model.py)"
  echo "lens/unified.py         md5 $(md5sum lens/unified.py | cut -d' ' -f1) mtime $(stat -c '%y' lens/unified.py)"
  echo "the recorded 15:35 hash was 64929ca223456f74e78a6c16ae8f5f9d; the file moved at 19:14:55, i.e. AFTER"
  echo "the 19:11 read this turn took and NOT by this turn (this turn edited lens/unified.py only). The new"
  echo "flags are --direct (store_true, default OFF) and --direct-mode (default lowrank, the 19:14 addition"
  echo "is the 'tied' option), so the arms of the in-flight 16-arm comparison are unaffected in behaviour;"
  echo "the queued opt_smoke.sh re-checks the default path digit for digit against the 15:35 prefix."
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/UNIFIED-LADDER.txt 2>&1
