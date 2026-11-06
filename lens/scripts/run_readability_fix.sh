#!/bin/bash
# READABILITY FIX - sweep the activation-sparsity dial, then judge each arm on the RANDOM gate.
#
# The failure being fixed (measured 2026-10-18, random draw of 200 held-out windows on sb_S0):
#   2334 false fires -- units firing with none of their own tokens present; only 17.2% of fires were
#   honest; 14.10 units fire per window out of 41. That is not an explanation, it is a fog.
# The dial that should fix it: lam_act, the penalty on mean(|concept activation|). It has been 0.0 in
# every run so far, so concepts are under no pressure to stay silent.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (one per arm):
#   lam_act 0.1  -> false fires fall below 1500, units/window below 11
#   lam_act 0.5  -> false fires below 700,  units/window below 8
#   lam_act 1.5  -> false fires below 300,  units/window below 5, and the risk is dead concepts
#   in every arm: counterfactual check stays 200/200 (the answer must still come from the named units),
#   and honest next-word accuracy must not collapse relative to the same-recipe arm without the penalty.
# Judge on the random gate, never on a hand-picked sample. Arms that finish are skipped on a re-run.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
OUT=lens/runs/READABILITY-FIX.txt
LOG=/tmp/readability_fix.log
exec > "$LOG" 2>&1

echo "=== READABILITY FIX SWEEP $(date '+%F %H:%M')" | tee -a "$OUT"
grep -q "## SHIP rx_lamact0.1_d256_24" "$OUT" || echo '## SHIP rx_lamact0.1_d256_24 -> lens/runs/chat/rx_lamact_d256_24_ep24.npz (this is arm lam_act=0.1; its filenames lost the value to the local-assignment bug)' >> "$OUT"
echo "one change per arm: lam_act. Everything else matches the turn-end arm recipe." | tee -a "$OUT"

# wait for the GPU to be quiet (the turn-end gate is running)
while pgrep -f "[t]rain_chat" > /dev/null; do sleep 60; done
echo "GPU quiet, starting $(date '+%H:%M')" | tee -a "$OUT"

run_arm () {
  local la="$1"
  local name="rx_lamact${la}_d256_24"   # separate line: ${la} is empty otherwise
  if grep -q "## SHIP $name" "$OUT" 2>/dev/null; then
    echo "## ARM $name: already finished and judged -- skipping" | tee -a "$OUT"; return
  fi
  echo "############################################################" | tee -a "$OUT"
  echo "## ARM $name  d=256 epochs=24 lam_act=$la  $(date '+%H:%M')" | tee -a "$OUT"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest" \
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
    --lam-act "$la" --seed 0 --device gpu --assistant-only --train-turn-end --max-windows 1500000 \
    --out "$name" 2>&1 | tee -a "$OUT.raw"
  local ship
  ship=$(ls -t lens/runs/chat/${name}_ep*.npz 2>/dev/null | head -1)
  echo "## SHIP $name -> $ship" | tee -a "$OUT"
  echo "## honest next-word accuracy:" | tee -a "$OUT"
  EVAL_DATA_DIR="$PWD/data/chat/prepared_ext_honest" bash lens/scripts/run_eval_clean.sh "$ship" 2>&1 \
    | grep -E "assistant-masked|plain pool" | tee -a "$OUT"
  echo "## RANDOM readability gate (the judge):" | tee -a "$OUT"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest" SHIP="$ship" \
    bash lens/scripts/run_random_readability.sh "$ship" 2>&1 \
    | grep -E "false fires|answer changes|explain exact|units firing|honest fires|READABILITY GATE" \
    | tee -a "$OUT"
}

run_arm 0.1
run_arm 0.5
run_arm 1.5
run_arm 3.0

echo "=== SWEEP DONE $(date '+%F %H:%M')" | tee -a "$OUT"