#!/bin/bash
# SMALL SMART MODEL on the COMBINED corpus (talk + maths + knowledge + code), standard chat format.
#
# This is the run the user asked for: a SMALL model (well under the 20M-parameter limit) that talks,
# codes and has some general knowledge, tested with random draws at every gate.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN:
#   P1 MMLU moves above the current 26.4% (chance 25%) -- the corpus now contains real knowledge text
#   P2 code syntax-valid moves above the current 4.0% -- the corpus now contains real code
#   P3 GSM8K stays near zero (1/200 today): a 4M-parameter model is not expected to do multi-step
#      arithmetic, and saying otherwise would be dishonest
#   P4 assistant-masked accuracy on held-out chat windows DROPS a little versus the chat-only model of
#      the same width, because the same capacity now covers maths, knowledge and code as well
#   P5 the readability gate still FAILS (nothing in this run changes how broadly concept units fire)
#   P6 P(<|im_end|>) at true turn boundaries is above zero at this size too
# Any prediction that fails is reported as failed, with the numbers that failed it.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
OUT=lens/runs/MIX-SMALL.txt
LOG=/tmp/mix_small.log
exec > "$LOG" 2>&1

echo "=== SMALL SMART MODEL on the COMBINED corpus $(date '+%F %H:%M')" | tee -a "$OUT"
echo "corpus data/chat/prepared_mix_ctrl_honest  format <|im_start|> role <text> <|im_end|>" | tee -a "$OUT"
echo "predictions: MMLU > 26.4%; code > 4.0%; GSM8K near zero; chat accuracy dips slightly;" | tee -a "$OUT"
echo "readability still FAILS; P(<|im_end|>) > 0" | tee -a "$OUT"

# wait for the corpus build and the GPU ladders
while pgrep -f "[b]uild_knowledge_corpus|[m]erge_corpora|[a]dd_control_tokens|[r]un_ladder_ctrl|[r]un_size_sweep" > /dev/null; do
  sleep 60
done
[ -f data/chat/prepared_mix_ctrl_honest/ids_v12000.npy ] || {
  echo "## ABORT: the combined corpus was not built -- nothing to train on" | tee -a "$OUT"; exit 1; }
echo "GPU and corpus ready, starting $(date '+%H:%M')" | tee -a "$OUT"

NAME=mx_d256_small_24
if ! grep -q "## SHIP .*$NAME" "$OUT" 2>/dev/null; then
  echo "--- training d=256 (about 4M parameters) on the combined corpus" | tee -a "$OUT"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
    --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end --max-windows 1500000 \
    --out "$NAME" 2>&1 | tee -a "$OUT.raw"
fi
SHIP=$(ls -t lens/runs/chat/${NAME}_ep*.npz 2>/dev/null | head -1)
[ -n "$SHIP" ] || { echo "## ABORT: no checkpoint produced" | tee -a "$OUT"; exit 1; }
echo "## SHIP $SHIP" | tee -a "$OUT"
.venv-gpu/bin/python3 -c "
import sys, os
sys.path.insert(0, '$PWD'); sys.path.insert(0, '$PWD/lens'); sys.path.insert(0, '$PWD/lens/scripts')
os.environ['CHAT_DATA_DIR'] = '$PWD/data/chat/prepared_mix_ctrl_honest'
import lens
n = lens.load('$SHIP', device='cpu')
print(f'## SIZE {sum(p.data.size for p in n.parameters()):,} params, {n.K} named concepts, '
      f'{len(json.load(open(\"$PWD/data/chat/prepared_mix_ctrl_honest/vocab_v12000.json\")))} vocab')
" 2>/dev/null | tee -a "$OUT"

echo "--- gate 0: gradient check" | tee -a "$OUT"
.venv-gpu/bin/python3 -u lens/scripts/gradcheck_width.py --width 256 2>&1 | tail -2 | tee -a "$OUT"

echo "--- PREDICTIONS P1/P2/P3: the smartness ruler (MMLU / GSM8K / code syntax-valid)" | tee -a "$OUT"
BENCH_SHIP="$SHIP" BENCH_VOCAB="$PWD/data/chat/prepared_mix_ctrl_honest/vocab_v12000.json" \
  BENCH_DEVICE=cpu nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/bench_external.py 2>&1 \
  | grep -iE "MMLU|GSM8K|code|chance|accuracy|correct" | tail -14 | tee -a "$OUT"

echo "--- PREDICTION P4: honest next-word accuracy on held-out chat windows" | tee -a "$OUT"
EVAL_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" bash lens/scripts/run_eval_clean.sh "$SHIP" 2>&1 \
  | grep -E "assistant-masked|plain pool|WARNING" | tee -a "$OUT"

echo "--- PREDICTION P6: does it predict the stop token where a turn really ends?" | tee -a "$OUT"
BOUNDARY_TOKEN="<|im_end|>" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/turnend_probe.py \
  "$SHIP" 2>&1 | grep -E "boundary positions|mean P|top-1|share above|VERDICT" | tee -a "$OUT"

echo "--- stop control through the shared decoder" | tee -a "$OUT"
SHIP="$SHIP" bash lens/scripts/run_stop_control_v2.sh "$SHIP" 2>&1 \
  | grep -E "handed the turn back|ended on a sentence|unique tokens|stopped by|rule-forced" | tee -a "$OUT"

echo "--- PREDICTION P5: random readability gate, three independent draws" | tee -a "$OUT"
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
bash lens/scripts/run_ship.sh "$SHIP" 2>&1 | tail -3 | tee -a "$OUT"

echo "=== DONE $(date '+%F %H:%M')" | tee -a "$OUT"