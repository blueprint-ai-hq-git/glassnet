#!/bin/bash
# CODE EXPERIMENT: vocab 6,000 -> 32,000 on the CODE-ONLY corpus (already on disk, no downloads).
#
# WHY THIS ONE: the authoritative harness shows 4/25 syntax and 23 of 25 prompts contain `<unk>`
# where the function NAME should be. Measured on the corpus:
#     vocab 6,000  : <unk> = 5.74% of tokens
#     vocab 32,000 : <unk> = 3.74% of tokens   (this corpus, verified: 7,885,017 tokens, 38,235 docs)
# So the hypothesis is concrete and testable: more identifier slots -> fewer <unk> in the signature
# -> higher syntax rate. Everything else held at the code1 recipe.
#
# PRE-REGISTERED GATES (fixed BEFORE running):
#   q1  syntax rate beats 4/25 on the authoritative harness (reexec_code.py, both prompt styles)
#   q2  the <unk> rate inside generated code falls below the 6k baseline's
#   q3  if q1 fails, the vocabulary was NOT the blocker -> report it and stop adding vocabulary
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CD_VOCAB=32000 CODE_DOC=code

echo "=== CODE: vocab 32000, code-only corpus, d=192, 16 epochs ==="
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 16 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --device gpu --out covocab32k --max-windows 1200000 2>&1 \
  | grep -E "^  ep |final|saved|EXEC|syntax|runs|Error|Traceback" | tail -30
echo "=== DONE ==="
