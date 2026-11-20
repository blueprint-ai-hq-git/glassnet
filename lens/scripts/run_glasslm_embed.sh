#!/bin/bash
# DOES A FACTORED EMBEDDING BEAT A FULL TOKEN TABLE AT FEWER PARAMETERS? (2026-11-20 16:22)
#
# THE PROBLEM, MEASURED: the first run had E = 768,128 of 836,352 parameters = 92%. A model that is 92% lookup table can only
# memorise; his goal ("each layer 10x smarter") cannot be reached while the layers hold 8% of the capacity.
#
# THE ONE CHANGE: --emb-rank 0 (full table) vs --emb-rank 16 (E = A @ B, 193,056 params instead of 768,128, i.e. 3.98x fewer).
# Both arms run THE CURRENT CODE, same seed, same windows, same steps, same everything else. The full-table arm is re-run rather
# than compared against the earlier 12.32% because the code has changed since (factored path added), and a comparison against a
# different code revision is not matched.
#
# GRADIENT CERTIFIED BEFORE LAUNCH: factored path numerical check, max ABSOLUTE error 6.559e-10 across A, B, P, lnf, Wq_0, W1_0.
# (The first version of that check used a relative-only criterion and reported a false 1.698e-06 failure - the entries it flagged
# have gradient magnitude ~5e-04, so the ratio blew up on an absolute error of 3e-10.)
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 factored r=16 held-out accuracy is at least 11.32% (within 1.0 point of the full-table arm) while using 3.98x fewer
#      embedding parameters.
#   P2 total parameters fall by at least 3x (full 836,352 -> factored about 261,280).
# FALSIFIER: factored accuracy below 10.32% (more than 2.0 below the full table) -> rank 16 cannot represent what the table does at
#   this budget. ONE retry is allowed at rank 32; if that also fails, the honest conclusion is that the table is doing real work and
#   the "fewer parameters, smarter layers" direction needs a different mechanism, not a bigger rank.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/GLASSLM-EMBED.txt
: > "$OUT"
echo "=== full table vs factored embedding, $(date '+%F %T') ===" | tee -a "$OUT"
for RANK in 0 16; do
  echo "--- emb_rank=$RANK ---" | tee -a "$OUT"
  env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=. nice -n 19 ionice -c3 taskset -c 6,7 \
    .venv-gpu/bin/python3 -u lens/glass_lm.py \
    --steps 600 --batch 16 --T 32 --d 64 --layers 2 --heads 4 --d-mlp 128 --windows 20000 --lr 0.05 \
    --emb-rank "$RANK" --log-every 150 \
    >> "lens/runs/chain_glasslm_emb$RANK.out" 2>&1
  RC=$?
  echo "  rc=$RC" | tee -a "$OUT"
  [ "$RC" -ne 0 ] && { echo "  ARM $RANK FAILED" | tee -a "$OUT"; exit 1; }
  grep -E 'model:|OUTPUT-TABLE SHARE|matched constant|HELD-OUT' "lens/runs/chain_glasslm_emb$RANK.out" | tail -4 | tee -a "$OUT"
done
echo "--- P1/P2 verdict: compare the two HELD-OUT lines and the two parameter counts above ---" | tee -a "$OUT"
echo "GLASSLM-EMBED COMPLETE" | tee -a "$OUT"
