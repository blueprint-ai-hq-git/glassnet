#!/bin/bash
# CHAT EXPERIMENT: assistant-only loss masking (one change from v12k).
#
# THE MECHANISM: the chat loss is computed over the whole window, so the model spends capacity
# learning to re-predict the HUMAN's text as well as its own replies. Masking the loss to assistant
# tokens (weight 0 on human targets) directs all capacity at the turn the model is judged on.
# MEASURED share: 71.7% of the corpus is assistant text, so this is a MODEST reweighting, not a
# dramatic one — said up front so the result is not oversold either way.
#
# Verified before training: the mask marks the right tokens (verified_assistant_mask.py shows the
# H/A pattern following real conversation turns; 71.7% assistant, inside the 5-95% sanity band).
#
# PRE-REGISTERED GATES (fixed BEFORE the run):
#   a1  unk rate on the shared prompts is LOWER than v12k's 3.2%
#   a2  bits/token stays within +0.30 of v12k's 6.61
#   a3  if a1 fails -> assistant-only masking does NOT improve reply quality; report it and do not
#       keep tuning the mask (no weight tweaking to chase the gate)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

echo "=== CHAT: assistant-only loss masking, d=128 W=16 V=12000, 24 epochs ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 128 --window 16 --epochs 24 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --device gpu --assistant-only --out asonly_v12000 \
  --max-windows 1500000 2>&1 | grep -E "^  ep |final VAL|saved|assistant-only|Error|Traceback" | tail -32
echo "=== DONE ==="
