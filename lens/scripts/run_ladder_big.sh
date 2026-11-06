#!/bin/bash
# BIG RUNGS of the ladder: d = 1536, 2048, 3072 -- the top of the user's 50M-parameter budget.
#
# Parameter counts (printed for real by the runner, not estimated):
#   d=1536 ~20.9M   d=2048 ~27.8M   d=3072 ~41.4M
# d=4096 would be about 55M and is OUT of budget, so it is not run.
#
# Corpus: data/chat/prepared_mix_ctrl_honest -- talk + maths + general knowledge + code, standard chat
# format. These rungs answer the user's question directly: as we scale up, do reliability and smartness
# hold, and does training stay problem-free?
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   B1 the smartness ruler improves with size: MMLU and code syntax-valid both higher than at d=256
#   B2 honest next-word accuracy on held-out windows improves with size
#   B3 the readability gate still FAILS at every big rung (broad concept firing is not a capacity problem)
#   B4 P(<|im_end|>) at true turn boundaries rises further with size, but free generation still rarely
#      hands the turn back while the text is word salad
#   B5 d=3072 is the first rung where a 6 GB card is genuinely tight: if it fails, the failure is
#      reported as a real limit with the actual error, not hidden
# A failed rung is logged with its error tail and the ladder continues to the next size.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
OUT=lens/runs/LADDER-BIG.txt
LOG=/tmp/ladder_big.log
exec > "$LOG" 2>&1

echo "=== BIG RUNGS $(date '+%F %H:%M')  d=1536 / 2048 / 3072 (about 21M / 28M / 41M parameters)" | tee -a "$OUT"
echo "corpus data/chat/prepared_mix_ctrl_honest  standard chat format  <=50M parameter budget" | tee -a "$OUT"
echo "predictions: smartness and accuracy improve with size; readability still FAILS; d=3072 may be the" | tee -a "$OUT"
echo "first real VRAM limit on a 6 GB card and would be reported as such" | tee -a "$OUT"

# serialize behind EVERY other GPU job, including the small combined-corpus run
while pgrep -f "[t]rain_chat|[r]un_size_sweep|[r]un_ladder_ctrl|[r]un_mix_small|[b]uild_knowledge_corpus" > /dev/null; do
  sleep 60
done
[ -f data/chat/prepared_mix_ctrl_honest/ids_v12000.npy ] || {
  echo "## ABORT: combined corpus missing" | tee -a "$OUT"; exit 1; }
echo "GPU quiet and corpus present, starting $(date '+%H:%M')" | tee -a "$OUT"

for D in 1536 2048 3072; do
  NAME="bg_d${D}_mix_24"
  if grep -q "## RUNG $D DONE" "$OUT" 2>/dev/null; then
    echo "## RUNG $D: already finished -- skipping" | tee -a "$OUT"; continue
  fi
  echo "############################################################" | tee -a "$OUT"
  echo "## RUNG d=$D   $(date '+%H:%M')" | tee -a "$OUT"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$OUT"

  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d "$D" --window 32 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
    --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end --max-windows 1500000 \
    --out "$NAME" > "$OUT.raw.$D" 2>&1
  RC=$?
  echo "training exit=$RC" | tee -a "$OUT"
  if [ "$RC" -ne 0 ]; then
    echo "## RUNG $D FAILED (exit $RC). Last lines of the real error:" | tee -a "$OUT"
    tail -6 "$OUT.raw.$D" | tee -a "$OUT"
    echo "## RUNG $D DONE (failed)" | tee -a "$OUT"
    continue
  fi
  tail -2 "$OUT.raw.$D" | tee -a "$OUT"

  SHIP=$(ls -t lens/runs/chat/${NAME}_ep*.npz 2>/dev/null | head -1)
  [ -n "$SHIP" ] || { echo "## RUNG $D: no checkpoint" | tee -a "$OUT"; continue; }
  echo "## SHIP $SHIP" | tee -a "$OUT"
  .venv-gpu/bin/python3 -c "
import sys, os, json
sys.path.insert(0, '$PWD'); sys.path.insert(0, '$PWD/lens'); sys.path.insert(0, '$PWD/lens/scripts')
os.environ['CHAT_DATA_DIR'] = '$PWD/data/chat/prepared_mix_ctrl_honest'
import lens
n = lens.load('$SHIP', device='cpu')
p = sum(x.data.size for x in n.parameters())
print(f'## SIZE d=$D measured: {p:,} parameters ({p/1e6:.1f}M), {n.K} concepts, budget 50M -> '
      f\"{'INSIDE' if p < 50e6 else 'OVER'}\")" 2>/dev/null | tee -a "$OUT"

  echo "--- smartness ruler (MMLU / GSM8K / code syntax-valid)" | tee -a "$OUT"
  BENCH_SHIP="$SHIP" BENCH_VOCAB="$PWD/data/chat/prepared_mix_ctrl_honest/vocab_v12000.json" \
    BENCH_DEVICE=cpu nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/bench_external.py 2>&1 \
    | grep -iE "MMLU|GSM8K|code|chance" | tail -12 | tee -a "$OUT"

  echo "--- honest next-word accuracy" | tee -a "$OUT"
  EVAL_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" bash lens/scripts/run_eval_clean.sh "$SHIP" 2>&1 \
    | grep -E "assistant-masked|plain pool|WARNING" | tee -a "$OUT"

  echo "--- stop-token probe at true turn boundaries" | tee -a "$OUT"
  BOUNDARY_TOKEN="<|im_end|>" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/turnend_probe.py \
    "$SHIP" 2>&1 | grep -E "boundary positions|mean P|top-1|share above|VERDICT" | tee -a "$OUT"

  echo "--- stop control through the shared decoder" | tee -a "$OUT"
  SHIP="$SHIP" bash lens/scripts/run_stop_control_v2.sh "$SHIP" 2>&1 \
    | grep -E "handed the turn back|ended on a sentence|unique tokens|stopped by" | tee -a "$OUT"

  echo "--- random readability gate, three independent draws" | tee -a "$OUT"
  for S in 20261018 7 12345; do
    echo "  (random draw seed $S)" | tee -a "$OUT"
    SHIP="$SHIP" N=200 SEED="$S" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
      lens/scripts/random_readability.py 2>&1 \
      | grep -E "false fires|answer changes|units firing|honest fires|READABILITY GATE" | tee -a "$OUT"
  done

  echo "--- text quality" | tee -a "$OUT"
  .venv-gpu/bin/python3 -u lens/scripts/talk_quality.py "$SHIP" "$NAME" 2>&1 \
    | grep -iE "this model|VERDICT" | tail -3 | tee -a "$OUT"

  echo "--- ship it: both standard containers" | tee -a "$OUT"
  bash lens/scripts/run_ship.sh "$SHIP" 2>&1 | tail -2 | tee -a "$OUT"

  echo "## RUNG $D DONE" | tee -a "$OUT"
done

echo "=== BIG RUNGS DONE $(date '+%F %H:%M')" | tee -a "$OUT"