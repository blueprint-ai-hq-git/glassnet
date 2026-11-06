#!/bin/bash
# H-BLOCK: the next experiments, all on the REPAIRED corpora so the in-training val is honest too.
# Waits for the running chains to finish first, so only one GPU job is ever active.
# Recorded plan: NEXT-EXPERIMENTS.md. Results append to lens/runs/H-BLOCK.txt.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/H-BLOCK.txt
LOG=/tmp/h_block.log
exec >> "$LOG" 2>&1

echo "=== waiting for the whole training queue to finish $(date '+%F %H:%M')"
# BUG FIXED 2026-10-17: this used to wait on "[r]un_full_schedule", which is not the name of the live
# wrapper (it is run_tomorrow.sh), so the block started immediately and two trainers shared the GPU.
# Now it waits until NO trainer and NO queue wrapper is alive.
while pgrep -f "[t]rain_chat" > /dev/null \
   || pgrep -f "[r]un_tomorrow|[r]un_small_big_queue|[r]un_corpus_ab|[r]un_full_schedule" > /dev/null; do
  sleep 60
done
echo "=== GPU quiet, starting H-block $(date '+%F %H:%M')"

run_arm () {
  local name="$1" d="$2" extra="$3" epochs="$4" corpus="$5"
  local data="$PWD/data/chat/$corpus"
  # RESUME: if this arm already has checkpoints (e.g. it was interrupted), continue from the newest one
  # instead of retraining from scratch. train_chat refuses a resume whose corpus differs, which is the
  # behaviour we want.
  local ck res=""
  ck=$(ls -t "lens/runs/chat/${name}_ep"*.npz 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    local done_ep
    done_ep=$(basename "$ck" .npz | sed 's/.*_ep//')
    res="--resume $ck --resume-ep $done_ep"
    echo "## ARM $name: resuming from $ck (ep $done_ep of $epochs)" | tee -a "$OUT"
  fi
  echo "############################################################" | tee -a "$OUT"
  echo "## ARM $name  d=$d extra='${extra:-none}' epochs=$epochs corpus=$corpus  $(date '+%H:%M')" | tee -a "$OUT"
  CHAT_DATA_DIR="$data" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d "$d" --window 32 --epochs "$epochs" --batch 256 --lr 0.4 --lr-decay 0.5 \
    --seed 0 --device gpu --assistant-only --max-windows 1500000 \
    $extra $res --out "$name" 2>&1 \
    | grep -E "window cap|assistant-only|ep 1/|ep 48/|ep 96/|ep 192/|final" | tee -a "$OUT"
  local ship="lens/runs/chat/${name}_ep${epochs}.npz"
  [ -f "$ship" ] || ship=$(ls -t lens/runs/chat/${name}_ep*.npz 2>/dev/null | head -1)
  echo "## SHIP $name -> $ship" | tee -a "$OUT"
  echo "## clean val (RAW and HONEST both printed):" | tee -a "$OUT"
  EVAL_DATA_DIR="$data" bash lens/scripts/run_eval_clean.sh "$ship" 2>&1 \
    | grep -E "SHIP|RAW|HONEST|assistant-masked" | tee -a "$OUT"
  echo "## talk quality:" | tee -a "$OUT"
  bash lens/scripts/run_talk_quality.sh "$ship" "$name" "$data" 2>&1 \
    | grep -E "trigram log-prob|this model|VERDICT" | tee -a "$OUT"
  echo "## concepts audited:" | tee -a "$OUT"
  grep -oE "concepts [0-9]+/[0-9]+" "$LOG" | tail -1 | tee -a "$OUT"
}

# NOTE: --lam-concept is only passed where it differs from the trainer's default of 0.3, so the
# pipeline's own default stays visible in every other arm's command line.
run_arm "h_H1_merged_honest_d256_96"    256 ""                          96  prepared_ext_honest
run_arm "h_H2_merged_honest_noConcept"  256 "--lam-concept 0.0"         96  prepared_ext_honest
run_arm "h_H3_merged_honest_w64"        256 "--window 64"               96  prepared_ext_honest
run_arm "h_H4_merged_honest_d256_192"   256 ""                          192 prepared_ext_honest
run_arm "h_H5_merged_honest_d512_192"   512 ""                          192 prepared_ext_honest

echo "=== H-BLOCK DONE $(date '+%F %H:%M')" | tee -a "$OUT"