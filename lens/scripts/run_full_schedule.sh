#!/bin/bash
# FULL-SCHEDULE ARMS — the push to actually REACH the recorded numbers on our current corpus.
#
# Yahya's direction (2026-10-16): stop explaining why the numbers differ and go get the performance.
# Targets, measured with the project's clean-val protocol on each run's own val split:
#   small model (d256): match or beat the recorded 54.89%
#   big model   (d512): match or beat the recorded 63.85%
#
# WHY 192 EPOCHS: every recorded number came from a 192-epoch run. Our 96-epoch arms are half the old
# schedule, so comparing them to the records mixes two changes at once. These arms use the FULL schedule.
#
# ARMS, best information first:
#   F1  merged  d256 192ep   does the SMALL model reach the target on the corpus we have now?
#   F2  splitfix d512 192ep  known-answer CONTROL: this exact recipe produced the recorded 63.85%.
#                            If it does not reproduce, the pipeline changed and every current number
#                            is suspect -- that is why it is here, not later.
#   F3  merged  d512 192ep   the big model at full schedule on the current corpus.
#   F4  splitfix d256 192ep  small-model control on the corpus that has a recorded small-model number.
#
# PRE-REGISTERED BARS
#  * primary: clean-val assistant-masked accuracy on the run's own val split
#  * talk quality: v2 meter must land ABOVE the shuffled/real midpoint for the run to count as "talking"
#  * the trainer's own val number is now the assistant-masked one; it must agree with run_eval_clean.sh
#    (same definition) or the disagreement is itself a finding to report before any number is quoted
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/FULL-SCHEDULE.txt
LOG=/tmp/full_schedule.log
exec >> "$LOG" 2>&1

echo "=== waiting for the corpus arms to finish $(date '+%F %H:%M')"
while pgrep -f "[r]un_corpus_ab.sh" > /dev/null; do sleep 60; done
echo "=== starting full-schedule arms $(date '+%F %H:%M')"

run_arm () {
  local name="$1" d="$2" corpus="$3" epochs="$4"
  local data="$PWD/data/chat/$corpus"
  # SKIP / RESUME (added 2026-10-17 when the user asked to stop and save): an arm whose eval is already
  # logged is finished, so never re-run it; an arm with checkpoints but no eval is resumed from its
  # newest checkpoint instead of restarting. Without this, stopping mid-arm threw away hours of GPU time.
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
    echo "## ARM $name: resuming from epoch $done_ep of $epochs (not restarting)" | tee -a "$OUT"
  fi
  echo "############################################################" | tee -a "$OUT"
  echo "## ARM $name  d=$d corpus=$corpus epochs=$epochs  $(date '+%H:%M')" | tee -a "$OUT"
  CHAT_DATA_DIR="$data" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d "$d" --window 32 --epochs "$epochs" --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --seed 0 --device gpu --assistant-only --max-windows 1500000 \
    $res --out "$name" 2>&1 | grep -E "window cap|windows |assistant-only|ep 1/|ep 48/|ep 96/|ep 144/|ep ${epochs}/|final" | tee -a "$OUT"
  local ship="lens/runs/chat/${name}_ep${epochs}.npz"
  [ -f "$ship" ] || ship=$(ls -t lens/runs/chat/${name}_ep*.npz 2>/dev/null | head -1)
  echo "## SHIP $name -> $ship" | tee -a "$OUT"
  echo "## clean val (own corpus):" | tee -a "$OUT"
  EVAL_DATA_DIR="$data" bash lens/scripts/run_eval_clean.sh "$ship" 2>&1 \
    | grep -E "assistant-masked|plain pool|SHIP" | tee -a "$OUT"
  echo "## talk quality:" | tee -a "$OUT"
  bash lens/scripts/run_talk_quality.sh "$ship" "$name" "$data" 2>&1 \
    | grep -E "trigram log-prob|this model|VERDICT" | tee -a "$OUT"
}

run_arm "fs_F1_merged_d256_192"    256 prepared_ext     192
run_arm "fs_F2_splitfix_d512_192"  512 prepared_splitfix 192
run_arm "fs_F3_merged_d512_192"    512 prepared_ext     192
run_arm "fs_F4_splitfix_d256_192"  256 prepared_splitfix 192

echo "=== FULL-SCHEDULE ARMS DONE $(date '+%F %H:%M')" | tee -a "$OUT"
