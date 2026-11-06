#!/bin/bash
# OVERNIGHT SUPERVISOR — the entire remaining talk-line queue, in the right order, DETACHED.
#
# Why this exists: the previous queue was three separate waiters launched as Hermes-session
# background jobs, and one of them had a wait condition that was already false, so the comparison
# matrix started IMMEDIATELY while the attention run was still training -- two trainers on one GPU
# (attention 902 MiB + matrix 454 MiB), which both slows every run and contaminates the attention
# run's epoch timing (the number that was supposed to prove the bmm3 speed fix). Session-tied jobs
# also die if the Hermes session ends, which would strand an overnight queue.
#
# This script runs as its own session (setsid from the launcher) and executes, strictly in order:
#   1. wait for the attention trainer that is currently running
#   2. measure + honesty-audit its ship
#   3. the merged-corpus 192-epoch reference run
#   4. measure + honesty-audit
#   5. the six-option comparison matrix (M0..M5), each measured on the common yardstick + audited
#   6. rank them and name the winner on evidence
#   7. append a night report to LOG.md
# It never touches anything outside the GlassNet folder and never deletes an artifact.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=/tmp/overnight.log
exec >> "$OUT" 2>&1
echo "=== SUPERVISOR START $(date '+%F %H:%M') (pid $$, session $(ps -o sid= -p $$ | tr -d ' '))"

# ---------------------------------------------------------------- 1 wait for the attention run
while pgrep -f "[t]rain_chat.py .*--out t12_attn_d512" >/dev/null 2>&1; do sleep 60; done
echo "=== attention run finished $(date '+%H:%M')"

# ---------------------------------------------------------------- 2 measure + audit it
ATT=$(ls -t lens/runs/chat/t12_attn_d512_ep*.npz 2>/dev/null | head -1)
if [ -n "$ATT" ]; then
  echo "--- attention ship: $ATT"
  EVAL_DATA_DIR="$PWD/data/chat/prepared_ext" bash lens/scripts/run_eval_clean.sh "$ATT" 2>&1 | tail -4
  EVAL_DATA_DIR="$PWD/data/chat/prepared_splitfix" bash lens/scripts/run_eval_clean.sh "$ATT" 2>&1 | tail -2
  TALK_CKPT="$(basename "$ATT")" .venv-gpu/bin/python3 -u lens/scripts/audit_strict_text.py 2>&1 \
      | grep -E "STRICT LABEL|s1\[talk\]" | tail -3
fi

# ---------------------------------------------------------------- 3 merged-corpus reference run
echo "=== merged-192 starting $(date '+%H:%M')"
bash lens/scripts/run_t12_merged.sh > /tmp/t12_merged.log 2>&1
echo "=== merged-192 finished $(date '+%H:%M')"
MERGED_SHIP=$(ls -t lens/runs/chat/t12_merged_d512_ep*.npz 2>/dev/null | head -1)
if [ -n "$MERGED_SHIP" ]; then
  echo "--- merged ship: $MERGED_SHIP"
  EVAL_DATA_DIR="$PWD/data/chat/prepared_ext" bash lens/scripts/run_eval_clean.sh "$MERGED_SHIP" 2>&1 | tail -4
  TALK_CKPT="$(basename "$MERGED_SHIP")" .venv-gpu/bin/python3 -u lens/scripts/audit_strict_text.py 2>&1 \
      | grep -E "STRICT LABEL|s1\[talk\]" | tail -3
fi

# ---------------------------------------------------------------- 5 the six-option matrix
echo "=== matrix starting $(date '+%H:%M')"
bash lens/scripts/run_matrix.sh
echo "=== matrix finished $(date '+%H:%M')"

# ---------------------------------------------------------------- 6 rank + name the winner
.venv-gpu/bin/python3 -u lens/scripts/matrix_summary.py

# ---------------------------------------------------------------- 7 night report into LOG.md
{
  echo ""
  echo "### OVERNIGHT QUEUE REPORT ($(date '+%F %H:%M'))"
  echo '```'
  grep -E "^##########|^(M[0-9]|  \(|  \(|  \(C\))|assistant-masked|STRICT LABEL|s1\[talk\]|WINNER|reference:" /tmp/matrix.log 2>/dev/null | head -60
  echo '```'
  echo "Full logs: /tmp/overnight.log, /tmp/t12_merged.log, /tmp/matrix.log, lens/runs/MATRIX.txt"
} >> LOG.md
echo "=== SUPERVISOR DONE $(date '+%F %H:%M')"