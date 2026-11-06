#!/bin/bash
# run_sentstate_wire_test.sh -- run the STRUCTURAL test of the sentstate graft and APPEND its output to
# lens/runs/SENTSTATE-WIRE.txt. CPU only, a few seconds, no corpus, no GPU, no checkpoint written outside a
# temp dir. Predictions: lens/runs/SENTSTATE-WIRE-PREDICTIONS-P114.txt (written before the test existed).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
OUT=lens/runs/SENTSTATE-WIRE.txt
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

{
  echo ""
  echo "=== STRUCTURAL TEST (lens/scripts/sentstate_wire_test.py) $(date '+%F %H:%M:%S')"
  echo "    predictions: lens/runs/SENTSTATE-WIRE-PREDICTIONS-P114.txt"
} >> "$OUT"

nice -n 19 .venv-gpu/bin/python3 -u lens/scripts/sentstate_wire_test.py >> "$OUT" 2>&1
RC=$?
echo "TEST EXIT=$RC" >> "$OUT"
grep -E "VERDICT|RESULT|PASS|FAIL" "$OUT" | tail -25
exit $RC
