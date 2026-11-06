#!/bin/bash
# Does ATTENTION earn its cost at the best known setting? d=256, 12 epochs, clean corpus.
# This also produces the first CLEAN ship that HAS an attention block, which the KV cache needs in order to be
# wired into decoding at all (every recent clean model is no-attention, so the cache has nothing to hook into).
#
# PREDICTION, written BEFORE the run (required rule):
#   the honest masked val will land within about 1 point of the 28.30% no-attention result at the same setting.
#   Attention is expected to help little at a 32-token window, because the concept layer already summarises it.
#   Above 29.3% attention earns its cost and stays; below 27.3% it is dropped from the clean recipe and kept
#   only for long-context work, where the KV cache makes it pay.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== ATTENTION AT THE BEST SETTING (d=256, 12 epochs, clean corpus) $(date '+%F %H:%M')"
  echo "comparison baseline: no-attention d=256/12ep = 28.30% final, peak 28.52% at ep9 (clean val)"
  echo "prediction: within ~1 point of 28.30%; above 29.3% keeps attention, below 27.3% drops it"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 12 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end --attention \
    --max-windows 1500000 --out cl_d256_attn_12
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/ATTN-BEST.txt 2>&1
