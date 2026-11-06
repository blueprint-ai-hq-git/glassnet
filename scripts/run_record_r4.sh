#!/bin/bash
# run_record_r4.sh — record the R4 result + caveats in LOG.md and the three ideas in
# QUEUE-AFTER-B.md. Touches nothing else: not the cards, not the key, not the rules, not B.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/record_r4.py 2>&1 | tail -4
echo "--- verify nothing else was touched ---"
ls -la runs/t3/reader_test_key_c.json READER-TEST-CARDS-C.md TARGET-B.md
echo "--- B untouched: checkpoint count and grid state ---"
ls -1 runs/t3/ckpt/ | grep -c "_b_seed" || true
grep -c "^--- " runs/t3/b_run.log 2>/dev/null || echo "b_run.log: no lines"