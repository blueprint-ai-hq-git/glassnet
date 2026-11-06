#!/bin/bash
# MORNING SUPERVISOR — resumes the talk-line work after the shutdown, in order of information value.
#
# 1. finish the attention run (12 epochs left) and MEASURE it -- the token-order lever, the first
#    real test of whether word salad was caused by having no order information. The run was killed at
#    ep180 when Yahya powered down; it resumes from that checkpoint.
# 2. the six-option comparison matrix (M0..M5) -- which corpus / attention / capacity combination
#    actually wins, all measured on ONE common held-out val. Resume-aware: the half-finished M0 run
#    from last night continues from its own checkpoint instead of restarting.
# 3. the merged-corpus 192-epoch reference run last (lowest information value of the three).
# Every step is resumable, so a shutdown at any point costs at most one epoch.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
exec >> /tmp/morning.log 2>&1
echo ""
echo "=== MORNING SUPERVISOR START $(date '+%F %H:%M') (pid $$, session $(ps -o sid= -p $$ | tr -d ' '))"

# ---------------------------------------------------------------- 1 attention: resume + measure
echo "=== [1/3] resuming the attention run from its newest checkpoint $(date '+%H:%M')"
ATT_LATEST=$(ls -t lens/runs/chat/t12_attn_d512_ep*.npz 2>/dev/null | head -1)
ATT_EP=$(basename "$ATT_LATEST" | sed -E 's/.*_ep([0-9]+)\.npz/\1/')
if [ "$ATT_EP" -lt 192 ]; then
  # the attention run belongs to the CURATED corpus lineage (run_t12_attn.sh sets
  # data/chat/prepared_clean). Resuming it against a different corpus rebuilt a 40-concept layer
  # against its 41-concept checkpoint and the loader correctly refused.
  CHAT_DATA_DIR="$PWD/data/chat/prepared_clean" nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 512 --window 32 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
    --seed 0 --device gpu --assistant-only --attention --max-windows 1500000 \
    --resume "$ATT_LATEST" --resume-ep "$ATT_EP" --out t12_attn_d512 2>&1 | tail -4
fi
ATT=$(ls -t lens/runs/chat/t12_attn_d512_ep*.npz | head -1)
echo "--- attention ship: $(basename "$ATT")"
EVAL_DATA_DIR="$PWD/data/chat/prepared_ext" bash lens/scripts/run_eval_clean.sh "$ATT" 2>&1 | tail -3
EVAL_DATA_DIR="$PWD/data/chat/prepared_splitfix" bash lens/scripts/run_eval_clean.sh "$ATT" 2>&1 | tail -2
TALK_CKPT="$(basename "$ATT")" .venv-gpu/bin/python3 -u lens/scripts/audit_strict_text.py 2>&1 \
    | grep -E "STRICT LABEL|s1\[talk\]" | tail -2

# ---------------------------------------------------------------- 2 the six-option matrix
echo "=== [2/3] six-option matrix starting $(date '+%H:%M')"
bash lens/scripts/run_matrix.sh
echo "=== matrix finished $(date '+%H:%M')"
.venv-gpu/bin/python3 -u lens/scripts/matrix_summary.py 2>&1 | tail -25

# ---------------------------------------------------------------- 3 merged-192 reference, last
echo "=== [3/3] merged-192 reference run starting $(date '+%H:%M')"
MERGED_LATEST=$(ls -t lens/runs/chat/t12_merged_d512_ep*.npz 2>/dev/null | head -1)
MERGED_ARGS=""
if [ -n "$MERGED_LATEST" ]; then
  MEP=$(basename "$MERGED_LATEST" | sed -E 's/.*_ep([0-9]+)\.npz/\1/')
  [ "$MEP" -lt 192 ] && MERGED_ARGS="--resume $MERGED_LATEST --resume-ep $MEP"
fi
CHAT_DATA_DIR="$PWD/data/chat/prepared_ext" nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 512 --window 32 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
  --seed 0 --device gpu --assistant-only --max-windows 1500000 $MERGED_ARGS \
  --out t12_merged_d512 2>&1 | tail -4
MSHIP=$(ls -t lens/runs/chat/t12_merged_d512_ep*.npz | head -1)
echo "--- merged ship: $(basename "$MSHIP")"
EVAL_DATA_DIR="$PWD/data/chat/prepared_ext" bash lens/scripts/run_eval_clean.sh "$MSHIP" 2>&1 | tail -3
TALK_CKPT="$(basename "$MSHIP")" .venv-gpu/bin/python3 -u lens/scripts/audit_strict_text.py 2>&1 \
    | grep -E "STRICT LABEL|s1\[talk\]" | tail -2

# ---------------------------------------------------------------- report into LOG.md
{
  echo ""
  echo "### MORNING PIPELINE REPORT ($(date '+%F %H:%M'))"
  echo '```'
  grep -E "attention ship|assistant-masked|STRICT LABEL|s1\[talk\]|^##########|SHIP |WINNER|reference:|variant|^  M[0-9]" /tmp/morning.log | head -60
  echo '```'
  echo "Raw: /tmp/morning.log, lens/runs/MATRIX.txt, lens/runs/MATRIX_SUMMARY.md"
} >> LOG.md
echo "=== MORNING SUPERVISOR DONE $(date '+%F %H:%M')"