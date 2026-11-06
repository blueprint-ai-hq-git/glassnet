#!/bin/bash
# CODE EXPERIMENT 4: CAPACITY. All three other levers (vocab, constrained decoding, data content)
# were tested and NONE reached significance. The consistent-direction pattern (20 -> 26 -> 30) is
# what a capacity limit looks like, so test capacity directly.
#
# ONE CHANGE: d 192 -> 384 (and the model gets ~4x bigger). Everything else identical to cclean32k:
# the clean corpus, vocab 32000, window 16, 16 epochs, lr 0.4 decay 0.5, same decoder, same eval.
#
# PRE-REGISTERED GATES (fixed BEFORE the run):
#   c1  syntax at n=150 beats the clean model's 30/150 with Fisher exact p < 0.05
#   c2  if c1 passes, capacity WAS the limit and the curve is documented for the next step
#   c3  if c1 fails, code generation at this scale is not capacity-limited either, and the honest
#       conclusion is that this architecture/data cannot do reliable code at any size tested.
#       Report that and STOP — no more code arms.
#
# VRAM CHECK: d=384, V=32000 -> output layer 384*32000*4B = 49MB, embeddings the same, K=38 concepts.
# Total well under the 5GB budget. Checked before launching.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CD_VOCAB=32000 CODE_DOC=clean

echo "waiting for the GPU to be free..."
while pgrep -f "train_code.py|train_chat.py" > /dev/null; do sleep 20; done
echo "GPU free; training a 4x WIDER model on the clean corpus"

nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 384 --window 16 --epochs 16 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --device gpu --out cbig384 --max-windows 1200000 2>&1 \
  | grep -E "^  ep |final|saved|params|Error|Traceback" | tail -30
echo "=== DONE ==="
