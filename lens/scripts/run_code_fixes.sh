#!/bin/bash
# run_code_fixes.sh — the two code fixes, ONE AT A TIME, each train + executable test.
#
# Measured diagnosis (diag_code.py, measure_vocab_need.py):
#   corpus built as problem + solution:
#       most common token  '`'      6.08%   (markdown backtick)
#       second             '<unk>'  5.74%   (1,119,649 unknown tokens, where identifiers go)
#       third              'the'    5.55%   (English prose)
#       43% of documents open with prose
#   on a real 322-token solution, <unk> covers 8.1% of tokens at 6k vocab, 0.3% at 32k
#
# CHANGE 1 (this run): DOC_MODE=code -- the training document is the SOLUTION ONLY.
#   measured effect on the corpus alone: prose-doc openings 43.0% -> 1.0%
#   BUT <unk> went 5.74% -> 7.21%, so this change alone is NOT expected to fix executability.
#   Pre-registered prediction: generated text stops being prose. Syntax rate improves from 4/25
#   but stays LOW, because <unk> is still the most common token.
#
# CHANGE 2 (next run, only if change 1 confirms the diagnosis): vocab 6k -> 32k.
#   Pre-registered: this is the one that should move executability.
#
# Step 1 is a SMALL rung first: 4 epochs, then the executable test, before any long run.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CODE_DOC=code

echo "############ SMALL RUNG: code-only corpus, 6k vocab, 4 epochs ############"
CODE_DOC=code nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs "${CE:-4}" --batch 128 --lr 0.4 --lam-concept 0.3 \
  --device cpu --max-windows "${CW:-120000}" --n-exec 8 --out codedoc 2>&1 | tail -30

echo
echo "############ THE EXECUTABLE TEST — did the output stop being prose? ############"
CKPT=codedoc_ep${CE:-4} CODE_DOC=code nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
  lens/scripts/reexec_code.py 2>&1 | tail -50
