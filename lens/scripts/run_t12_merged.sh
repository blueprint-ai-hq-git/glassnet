#!/bin/bash
# T12-MERGED-D512 — the DATA lever, with the Wi-Fi window's external corpus.
# ONE change vs the splitfix run (t12_splitfix_d512, the current best: 63.85% / ppl 85.4):
# the corpus. Same d512, same window, same lr/epochs/seed, same no-attention architecture.
# Corpus: data/chat/prepared_ext = HH-RLHF 13.3M + oasst1 7.5M + ultrachat 12.0M + orca-math 6.0M
# = 38.8M tokens (2.9x), leak-fixed at build time (867 val convs moved out of val).
# Bars (pre-registered, binary):
#   A) merged-corpus clean val: beat 63.85% assistant-masked   (same model, more data)
#   B) the OLD curated val (prepared_clean): report the number -- if it drops hard, the model
#      traded its original domain away, and that is a real finding, not a win.
#   C) strict audit 40/40 must hold.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
set -e
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ext"
echo "=== training on the merged corpus $(date +%H:%M)"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 512 --window 32 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only --max-windows 1500000 \
  --out t12_merged_d512 2>&1 | tail -25
echo "=== A) merged-corpus clean val:"
EVAL_DATA_DIR="$PWD/data/chat/prepared_ext" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/t12_merged_d512_ep192.npz
echo "=== B) OLD curated val (domain retention check):"
bash lens/scripts/run_eval_clean.sh lens/runs/chat/t12_merged_d512_ep192.npz
echo "=== C) strict audit on the new ship:"
TALK_CKPT=t12_merged_d512_ep192.npz .venv-gpu/bin/python3 -u lens/scripts/audit_strict_text.py \
  2>&1 | grep -E "STRICT LABEL|s1\[talk\]|s1\[code\]"