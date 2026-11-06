#!/bin/bash
# T12-CLEAN-D512 — the winner's architecture trained on the CURATED corpus (deduplicated).
# ONE change vs run_t12_d512_full.sh: the data dir (CHAT_DATA_DIR=prepared_clean).
# Bar (pre-registered, binary): on the CLEAN val split it must beat the raw-trained d512-full's
# clean-val number AND keep strict talk 40/40. If it does not, the honest conclusion is that the
# duplicated content was not what the model was learning from, and that gets recorded as such.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CHAT_DATA_DIR="$PWD/data/chat/prepared_clean"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 512 --window 32 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only --max-windows 1500000 \
  --out t12_clean_d512 2>&1 | tail -30
echo "=== clean-val eval:"
bash lens/scripts/run_eval_clean.sh lens/runs/chat/t12_clean_d512_ep192.npz
echo "=== audit:"
TALK_CKPT=t12_clean_d512_ep192.npz .venv-gpu/bin/python3 -u lens/scripts/audit_strict_text.py 2>&1 | grep -E "s1\[talk\]|mislabelled|STRICT LABEL" | head -3