#!/bin/bash
# CONTROL FOR THE CKPT_LIST GUARD (predictions, written BEFORE this file ran, inside this header):
#   C1  a GOOD spec (two items, both with '=') -> "CKPT_LIST guard: 2 item(s)" + "PASS" + exit 0
#   C2  the BAD spec that actually killed the self-context job's chained decider — a LABEL containing ';' —
#       -> "REFUSING" + exit 3, which in run_h2_noconcept.sh / run_h5l_d256_6ep.sh means the instrument is NOT
#       started and the runner logs "CKPT_LIST GUARD REFUSED".
# It needs no corpus, no model and no GPU. Falsified if either case behaves the other way round.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
LOG=lens/runs/CKPTLIST-GUARD-TEST.txt
{
  echo "===================================================================================================="
  echo "CKPT_LIST GUARD CONTROL  $(date '+%F %H:%M:%S')"
  echo "the guard as it now stands in run_h2_noconcept.sh / run_h5l_d256_6ep.sh, verbatim"
  echo "===================================================================================================="
  run_guard() {
    .venv-gpu/bin/python3 -c "
import os,sys
items=os.environ['CKPT_LIST'].split(';')
bad=[i for i in items if '=' not in i]
print('CKPT_LIST guard: %d item(s)'%len(items))
if bad:
    print('REFUSING: item(s) with no = (a label may not contain the ; separator): %r'%bad)
    sys.exit(3)
print('CKPT_LIST guard: PASS - every item carries a path')
"
  }

  echo "--- C1 GOOD spec (expect 2 items, PASS, exit 0) ---"
  CKPT_LIST="ARM best (one flag off the twin)=lens/runs/chat/h2_noconcept_169_3_best.npz;TWIN ct_d256_full169_fix3 (stored control)=lens/runs/chat/ct_d256_full169_fix3_best.npz" run_guard
  echo "C1 EXIT=$?  (expected 0)"

  echo "--- C2 BAD spec, the exact shape that fired: a label containing ';' (frac 0.5, rounds 4; held-out ...) ---"
  CKPT_LIST="SELF-CONTEXT arm (frac 0.5, rounds 4; held-out by position as logged above)=lens/runs/chat/sctx_d256_full169_best.npz;TWIN ct_d256_full169_fix3 (the matched control, stored, held-out 35.26% by position)=lens/runs/chat/ct_d256_full169_fix3_best.npz" run_guard
  echo "C2 EXIT=$?  (expected 3)"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> "$LOG" 2>&1
tail -n 20 "$LOG"
