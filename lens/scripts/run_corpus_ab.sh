#!/bin/bash
# FOLLOW-UP: isolate the CORPUS (the small model just scored 13.81% on the merged corpus at 96 epochs,
# while the reference scored 63.85% on the splitfix corpus at 192 epochs -- two differences at once).
# These two arms train on the SPLITFIX corpus with the SAME flags as S0/B0 (96 epochs, seed 0), so the
# only thing that changes versus them is the corpus. That is the one-change comparison.
#
# PRE-REGISTERED BARS
#  * corpus is the only change versus sb_S0_d256_noattn (13.81%) and sb_B0_d512_noattn
#  * measured with the same metric on each corpus's OWN val split (vocabularies differ, so numbers are
#    NOT comparable ACROSS corpora; the comparison is "does the merged corpus train far worse at equal
#    epochs", and each number is quoted with its own corpus)
#  * talk quality with the v2 meter, same bar (above the shuffled/real midpoint)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/SMALL-BIG.txt
LOG=/tmp/corpus_ab.log
exec >> "$LOG" 2>&1

echo "=== waiting for the small->big queue to finish $(date '+%F %H:%M')"
while pgrep -f "[r]un_small_big_queue.sh" > /dev/null; do sleep 60; done
echo "=== queue finished, starting the corpus arms $(date '+%F %H:%M')"

run_arm () {
  local name="$1" d="$2" corpus="$3"
  local data="$PWD/data/chat/$corpus"
  # SKIP / RESUME (added 2026-10-17, same rule as run_full_schedule.sh): finished arms are never
  # re-run; a partially trained arm resumes from its newest checkpoint.
  if grep -q "## SHIP $name" "$OUT" 2>/dev/null; then
    echo "## ARM $name: already finished and evaluated -- skipping" | tee -a "$OUT"
    return
  fi
  local ck res=""
  ck=$(ls -t "lens/runs/chat/${name}_ep"*.npz 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    local done_ep
    done_ep=$(basename "$ck" .npz | sed 's/.*_ep//')
    res="--resume $ck --resume-ep $done_ep"
    echo "## ARM $name: resuming from epoch $done_ep of 96 (not restarting)" | tee -a "$OUT"
  fi
  echo "############################################################" | tee -a "$OUT"
  echo "## ARM $name   d=$d   corpus=$corpus    $(date '+%H:%M')" | tee -a "$OUT"
  CHAT_DATA_DIR="$data" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d "$d" --window 32 --epochs 96 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --seed 0 --device gpu --assistant-only --max-windows 1500000 \
    $res --out "$name" 2>&1 | grep -E "window cap|windows |ep 1/96|ep 48/96|ep 96/96|final" | tee -a "$OUT"
  local ship="lens/runs/chat/${name}_ep96.npz"
  [ -f "$ship" ] || ship=$(ls -t lens/runs/chat/${name}_ep*.npz 2>/dev/null | head -1)
  echo "## SHIP $name -> $ship" | tee -a "$OUT"
  echo "## own-corpus clean val:" | tee -a "$OUT"
  EVAL_DATA_DIR="$data" bash lens/scripts/run_eval_clean.sh "$ship" 2>&1 \
    | grep -E "assistant-masked|plain pool|SHIP" | tee -a "$OUT"
  echo "## talk quality (v2 meter, own corpus):" | tee -a "$OUT"
  bash lens/scripts/run_talk_quality.sh "$ship" "$name" "$data" 2>&1 \
    | grep -E "trigram log-prob|this model|VERDICT" | tee -a "$OUT"
}

run_arm "sb_S2_splitfix_d256_96ep" 256 prepared_splitfix
run_arm "sb_S3_splitfix_d512_96ep" 512 prepared_splitfix

echo "=== CORPUS ARMS DONE $(date '+%F %H:%M')" | tee -a "$OUT"
