#!/bin/bash
# T12-D512-FULL — the winning ladder member at the BASELINE'S OWN epoch count (192).
# Bar (pre-registered, binary): strict audit 40/40 AND trainer val_acc(192ep) > 30.94 (t11 at
# 192) AND assistant-masked audit acc vs the record's 80.59% reported beside it. One concept was
# thin-margin (39/40) at 96 epochs; the t11's own 40/40 appeared only at 192, so extension is the
# fair test, not a tuning loop. Fresh run, seed 0, identical recipe, checkpoints every epoch.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 512 --window 32 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only --max-windows 1500000 \
  --out t12_d512_full 2>&1 | tail -30
echo "=== audit:"
ls -t lens/runs/chat/t12_d512_full_ep*.npz | head -1
TALK_CKPT=$(basename $(ls -t lens/runs/chat/t12_d512_full_ep*.npz | head -1)) \
  .venv-gpu/bin/python3 -u lens/scripts/audit_strict_text.py > /tmp/audit_d512full.out 2>&1
grep -E "STRICT LABEL|s1\[talk\]|mislabelled" /tmp/audit_d512full.out | head -4