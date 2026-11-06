#!/bin/bash
# T12-ATTN-D512 — THE ARCHITECTURE LEVER, never tried on this lineage (measured: t11 AND
# t12_d512_full both ran --attention=False, i.e. a bag-of-window concept net with NO token order).
# That single fact explains the measurements we already have: 54-56% next-token accuracy (a bag
# predicts topical/next-word-frequency well) TOGETHER WITH word-salad generation (without order
# the model cannot produce grammar), measured by the scenario battery on t12_d512_full:
#   'hello how are you ?' -> 'against all each other would be called seasons are lifting yourself...'
# ONE change vs the clean-data run: --attention (causal self-attention before the concept layer).
# Same curated corpus, same d512, same 192 epochs, same seed, every checkpoint saved.
# Bar (pre-registered, binary): on the CLEAN val split beat 56.07% assistant-masked AND keep
# strict talk 40/40 AND the scenario battery must stop producing word salad (judged on the same
# 5 prompts, output saved for comparison). If grammar does not appear, that is recorded as such.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CHAT_DATA_DIR="$PWD/data/chat/prepared_clean"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 512 --window 32 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only --attention --max-windows 1500000 \
  --out t12_attn_d512 2>&1 | tail -30
echo "=== clean-val eval:"
bash lens/scripts/run_eval_clean.sh lens/runs/chat/t12_attn_d512_ep192.npz
echo "=== scenario battery (the coherence test):"
TALK_CKPT=t12_attn_d512_ep192.npz bash lens/scripts/run_talk_scenarios.sh t12_attn_d512_ep192.npz
echo "=== audit:"
TALK_CKPT=t12_attn_d512_ep192.npz .venv-gpu/bin/python3 -u lens/scripts/audit_strict_text.py 2>&1 | grep -E "s1\[talk\]|mislabelled|STRICT LABEL" | head -3