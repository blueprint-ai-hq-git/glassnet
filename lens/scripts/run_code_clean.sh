#!/bin/bash
# CODE EXPERIMENT: train on the CLEANED corpus (fences + prose stripped).
#
# One change from covocab32k: the training DATA (clean corpus instead of fence+prose corpus).
# Vocabulary 32000, d=192, W=16, 16 epochs, decoder and eval all identical — so a difference can
# only come from the data content.
#
# Corpus facts (measured): 30,485 docs, 4.3M tokens, backticks 6.08% -> 0.0911%, unk 4.21%.
#
# PRE-REGISTERED GATES (fixed BEFORE the run, in LOG.md):
#   p2  syntax at n=150 beats the 6k baseline's 20/150 with Fisher exact p < 0.05
#   p3  if p2 fails -> data content is not the blocker either; code generation is capacity-limited
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CD_VOCAB=32000 CODE_DOC=clean

echo "waiting for the GPU to be free..."
while pgrep -f "train_chat.py" > /dev/null; do sleep 20; done
echo "GPU free; training on the CLEAN corpus"

nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 16 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --device gpu --out cclean32k --max-windows 1200000 2>&1 \
  | grep -E "^  ep |final|saved|EXEC|syntax|runs|Error|Traceback" | tail -30
echo "=== DONE ==="
