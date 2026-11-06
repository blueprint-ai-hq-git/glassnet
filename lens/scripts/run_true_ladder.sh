#!/bin/bash
# THE TRUE LADDER: every existing checkpoint measured on the repaired (provably disjoint) val splits.
# Also repairs the ORIGINAL 'prepared' corpus so the oldest ships can be measured honestly too.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/HONEST-LADDER.txt
exec > /tmp/honest_ladder.log 2>&1

echo "=== repairing the original corpus too"
.venv-gpu/bin/python3 -u lens/scripts/repair_split.py "$PWD/data/chat/prepared" \
  "$PWD/data/chat/prepared_honest" 32

echo "=== HONEST LADDER $(date '+%F %H:%M')" | tee -a "$OUT"
show () { grep -E "SHIP|RAW|HONEST|assistant-masked" ; }

echo "########## t12_splitfix (recorded 63.85) on prepared_splitfix_honest" | tee -a "$OUT"
EVAL_DATA_DIR="$PWD/data/chat/prepared_splitfix_honest" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/t12_splitfix_d512_ep192.npz 2>&1 | show | tee -a "$OUT"

echo "########## t11_talk41 (recorded 54.89) on prepared_honest" | tee -a "$OUT"
EVAL_DATA_DIR="$PWD/data/chat/prepared_honest" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/t11_talk41_ep192.npz 2>&1 | show | tee -a "$OUT"

echo "########## t12_d512_full (recorded 56.07) on prepared_honest" | tee -a "$OUT"
EVAL_DATA_DIR="$PWD/data/chat/prepared_honest" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/t12_d512_full_ep192.npz 2>&1 | show | tee -a "$OUT"

echo "########## our current small arm S0 on prepared_ext_honest" | tee -a "$OUT"
EVAL_DATA_DIR="$PWD/data/chat/prepared_ext_honest" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/sb_S0_d256_noattn_ep96.npz 2>&1 | show | tee -a "$OUT"

echo "########## our current big arm B0 on prepared_ext_honest" | tee -a "$OUT"
EVAL_DATA_DIR="$PWD/data/chat/prepared_ext_honest" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/sb_B0_d512_noattn_ep96.npz 2>&1 | show | tee -a "$OUT"

echo "=== DONE $(date '+%F %H:%M')" | tee -a "$OUT"