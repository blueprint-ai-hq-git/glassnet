#!/usr/bin/env bash
# DECORRELATION ARMS: penalty on correlated concept activations, on top of the shipped operating point (53 concepts, lam_concept 1.0).
# Predictions: lens/runs/DECORR-PREDICTIONS.txt | gradient verified in tools/decorr_penalty.py before this ran.
set -uo pipefail
cd "${GLASSNET_ROOT:?set GLASSNET_ROOT to the project folder}" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/DECORR.txt
: > "$OUT"
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export TC_CONCEPTS=chat_concepts_wide60
SPLIT="$PWD/data/chat/prepared_mix_ctrl_honest/split_matched_v1.npy"
if pgrep -x cs2 >/dev/null; then echo "SKIP: cs2 is running" | tee -a "$OUT"; exit 0; fi
for LD in 0.05 0.20; do
  TAG="wide60_lam1.0_decorr${LD}_s0"
  echo "=== arm $TAG start $(date '+%H:%M')" | tee -a "$OUT"
  taskset -c 2,3 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 1024 --window 32 --epochs 18 --batch 512 --lr 0.4 --lr-decay 0.5 --lam-concept 1.0 --lam-decorr "$LD" --lam-act 0.0 \
    --seed 0 --device gpu --assistant-only --train-turn-end --max-windows 600000 --drop-unk --no-epoch-ckpts \
    --split-file "$SPLIT" --out "nodirect1024_$TAG" > "logs/exposure/nodirect1024_$TAG.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "ARM FAILED $TAG rc=$rc" | tee -a "$OUT"; tail -8 "logs/exposure/nodirect1024_$TAG.log" | tee -a "$OUT"; exit 1
  fi
  grep -E "concepts: [0-9]+ kept|final VAL|-> [0-9]+/[0-9]+ pass" "logs/exposure/nodirect1024_$TAG.log" | tail -3 | tee -a "$OUT"
done
echo "=== breadth and activation spread" | tee -a "$OUT"
taskset -c 0,1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/measure_activation_spread.py \
  nodirect1024_wide60_lam1.0_s0 nodirect1024_wide60_lam1.0_decorr0.05_s0 nodirect1024_wide60_lam1.0_decorr0.20_s0 2>&1 | tee -a "$OUT"
echo "=== does the measured pair actually separate? (D4)" | tee -a "$OUT"
for T in nodirect1024_wide60_lam1.0_s0 nodirect1024_wide60_lam1.0_decorr0.05_s0 nodirect1024_wide60_lam1.0_decorr0.20_s0; do
  if [ -f "lens/runs/chat/${T}_best.npz" ]; then
    taskset -c 0,1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u tools/chat_concept_entanglement.py \
      --ckpt "lens/runs/chat/${T}_best.npz" --windows 4000 --top 3 2>&1 | grep -E 'effective dimensionality|cp_certainty|cp_emotion' | tee -a "$OUT"
  fi
done
echo "DECORR COMPLETE" | tee -a "$OUT"
