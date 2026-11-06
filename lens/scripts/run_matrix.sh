#!/bin/bash
# COMPARISON MATRIX — equal budget (96 epochs, seed 0), four options, one common yardstick.
# Rationale: instead of one serial change at a time, train all four options and pick the winner on
# evidence (the project's standing rule). The shared 12,000-word vocab across both corpora means
# the merged corpus's held-out val is a VALID common yardstick for every ship here.
#
#   M0  merged corpus   d512  no-attention   (matrix baseline)
#   M1  merged corpus   d512  ATTENTION      (the token-order lever on better data)
#   M2  merged corpus   d768  no-attention   (capacity lever)
#   M3  chat-only       d512  no-attention   (does the math data help or hurt conversation?)
#
# Every run: train -> eval on (A) the merged clean val, (B) its own corpus's clean val ->
# strict honesty audit. Results appended to lens/runs/MATRIX.txt as they land.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
MERGED="$PWD/data/chat/prepared_ext"
CHATONLY="$PWD/data/chat/prepared_ext_chat"
BIGMIX="$PWD/data/chat/prepared_ext_big"
OUT=lens/runs/MATRIX.txt
: > "$OUT"

run() {   # name  data_dir  out_name  extra_args
  local name="$1" data="$2" outn="$3" extra="$4"
  local total=96
  echo "########## $name  $(date +%H:%M)" | tee -a "$OUT"
  # RESUME-AWARE: if a checkpoint for this run already exists, continue from it instead of
  # starting over. This is what makes a shutdown cost nothing (proved: resume is bit-identical
  # because the shuffle RNG state is restored alongside the parameters).
  local latest resume_args=""
  latest=$(ls -t lens/runs/chat/${outn}_ep*.npz 2>/dev/null | head -1)
  if [ -n "$latest" ]; then
    local ep
    ep=$(basename "$latest" | sed -E 's/.*_ep([0-9]+)\.npz/\1/')
    if [ "$ep" -ge "$total" ] && [ ! -f lens/runs/chat/${outn}_RESUME_FORCE ]; then
      echo "  $outn is already complete at epoch $ep -- measuring the existing artifact, not retraining" | tee -a "$OUT"
      record "$name" "${outn}_ep${ep}.npz" "$data"
      return
    fi
    if [ "$ep" -lt "$total" ]; then
      resume_args="--resume $latest --resume-ep $ep"
      echo "  RESUME: $outn continues from $(basename "$latest") at epoch $((ep+1))" | tee -a "$OUT"
    fi
  fi
  CHAT_DATA_DIR="$data" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d "${D:-512}" --window 32 --epochs "$total" --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --seed 0 --device gpu --assistant-only --max-windows 1500000 \
    $extra $resume_args --out "$outn" 2>&1 | tail -3 | tee -a "$OUT"
  local ship="lens/runs/chat/${outn}_ep${total}.npz"
  if [ ! -f "$ship" ]; then
    ship=$(ls -t lens/runs/chat/${outn}_ep*.npz 2>/dev/null | head -1)
    echo "  (full-length checkpoint absent; measuring newest $(basename "$ship" 2>/dev/null))" | tee -a "$OUT"
  fi
  record "$name" "$(basename "$ship")" "$data"
}

record() {   # name  ship_file  own_data_dir
  local name="$1" ship="$2" own="$3"
  local path="lens/runs/chat/$ship"
  if [ ! -f "$path" ]; then echo "  SHIP MISSING: $path" | tee -a "$OUT"; return; fi
  echo "  SHIP $ship" | tee -a "$OUT"
  echo "  (A) merged clean val:" | tee -a "$OUT"
  EVAL_DATA_DIR="$MERGED" bash lens/scripts/run_eval_clean.sh "$path" 2>&1 | tail -3 | tee -a "$OUT"
  echo "  (B) own-corpus clean val:" | tee -a "$OUT"
  EVAL_DATA_DIR="$own" bash lens/scripts/run_eval_clean.sh "$path" 2>&1 | tail -2 | tee -a "$OUT"
  echo "  (C) strict audit:" | tee -a "$OUT"
  TALK_CKPT="$ship" .venv-gpu/bin/python3 -u lens/scripts/audit_strict_text.py \
    2>&1 | grep -E "STRICT LABEL|s1\[talk\]" | tee -a "$OUT"
}

D=512 run "M0 merged d512 no-attn"      "$MERGED"   t12m0_merged_d512        ""
D=512 run "M1 merged d512 ATTENTION"    "$MERGED"   t12m1_merged_d512_attn   "--attention"
D=768 run "M2 merged d768 no-attn"      "$MERGED"   t12m2_merged_d768        ""
D=512 run "M3 chat-only d512 no-attn"   "$CHATONLY" t12m3_chat_d512          ""
D=512 run "M4 BIG-MIX d512 no-attn"     "$BIGMIX"   t12m4_bigmix_d512        ""
D=512 run "M5 big-mix d512 ATTENTION"   "$BIGMIX"   t12m5_bigmix_d512_attn   "--attention"
echo "########## MATRIX DONE $(date +%H:%M)" | tee -a "$OUT"