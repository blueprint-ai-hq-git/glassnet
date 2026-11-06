#!/bin/bash
# ONE COMMAND tomorrow: finish B1 from its checkpoint, then the corpus arms, then the 192-epoch arms.
# B1 is resumed (not restarted) using the checkpoint + its RNG sidecar, so at most one epoch is redone.
# The trainer REFUSES a resume whose corpus differs, which is why CHAT_DATA_DIR is set explicitly here.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ext"

CK=$(ls -t lens/runs/chat/sb_B1_d512_attn_ep*.npz 2>/dev/null | head -1)
EP=$(echo "$CK" | sed -E 's/.*_ep([0-9]+)\.npz/\1/')
if [ -n "$CK" ] && [ "$EP" -lt 96 ]; then
  echo "=== resuming B1 (d512 + attention) from ep$EP"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 512 --window 32 --epochs 96 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
    --seed 0 --device gpu --assistant-only --max-windows 1500000 --attention \
    --resume "$CK" --resume-ep "$EP" --out sb_B1_d512_attn 2>&1 \
    | grep -E "resume|ep 96/|final" | tee -a lens/runs/SMALL-BIG.txt
  echo "## SHIP sb_B1_d512_attn -> $CK" >> lens/runs/SMALL-BIG.txt
  echo "## (A) merged clean val:" >> lens/runs/SMALL-BIG.txt
  EVAL_DATA_DIR="$CHAT_DATA_DIR" bash lens/scripts/run_eval_clean.sh "$CK" 2>&1 \
    | grep -E "assistant-masked|SHIP" >> lens/runs/SMALL-BIG.txt
  echo "## (B) talk quality:" >> lens/runs/SMALL-BIG.txt
  bash lens/scripts/run_talk_quality.sh "$CK" sb_B1_d512_attn "$CHAT_DATA_DIR" 2>&1 \
    | grep -E "this model|VERDICT" >> lens/runs/SMALL-BIG.txt
else
  echo "=== B1 already at ep${EP:-?}; nothing to resume"
fi

echo "=== then the corpus-isolation arms, then the full-schedule arms"
bash lens/scripts/run_corpus_ab.sh
bash lens/scripts/run_full_schedule.sh

echo "=== ALL TOMORROW'S ARMS DONE $(date '+%F %H:%M')"