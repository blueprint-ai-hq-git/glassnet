#!/bin/bash
# RE-VERIFY the CODE conclusion ("all four levers closed -> capacity-limited") on the CORRECTED
# gradient. Every code arm trained with --lam-concept 0.3 and real concept supports, so the whole
# negative result rests on the gradient that was wrong by a factor of pos_weight for support != 0.5.
# The k1 K-transfer gate already FLIPPED from PASS to FAIL when re-run on the fixed engine, so this
# conclusion cannot be assumed to survive either.
#
# Same recipes (one change at a time, as originally), same corpora, same epochs; only the engine's
# concept gradient is now correct. Outputs get a _fixedgrad suffix; originals KEPT.
#   lever 1  VOCABULARY   6k -> 32k        (orig: 20/150 -> 26/150, p=0.42 NOT significant)
#   lever 2  DATA CONTENT clean corpus     (orig: 20/150 -> 30/150, p=0.16 NOT significant)
#   lever 3  CAPACITY     d 192 -> 384     (orig: 30/150 -> 24/150, p=0.45 NOT significant)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

echo "=== 1/3 CODE baseline 6k vocab (fixed gradient) ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 16 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --device gpu --out cfix_code1_6k --max-windows 1200000 2>&1 \
  | grep -E "^  ep |final|EXEC|syntax|Error|Traceback" | tail -4

echo "=== 2/3 CODE vocab 32k (fixed gradient) ==="
CD_VOCAB=32000 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 16 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --device gpu --out cfix_covocab32k --max-windows 1200000 2>&1 \
  | grep -E "^  ep |final|EXEC|syntax|Error|Traceback" | tail -4

echo "=== 3/3 CODE clean corpus 32k vocab (fixed gradient) ==="
CD_VOCAB=32000 CODE_DOC=clean nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 16 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --device gpu --out cfix_cclean32k --max-windows 1200000 2>&1 \
  | grep -E "^  ep |final|EXEC|syntax|Error|Traceback" | tail -4

echo "=== CODE FIXED-GRADIENT RE-VERIFICATION DONE ==="