#!/bin/bash
# run_code_real.sh — the real code model, judged by EXECUTION.
#
# The smoke verified the whole path: 38 concepts kept, 27:1 bottleneck, and the execution test
# actually ran 8 generated snippets and reported 0/8 (correct for 1 epoch on 6,000 windows).
#
# Pre-registered predictions (written BEFORE the run):
#   1. Perplexity falls far below the smoke's 1112 (which was 1 epoch on 6,000 windows).
#   2. SYNTACTIC validity rises well above the smoke's 0/8 -- this is the first thing a small code
#      model learns, because indentation and `def ...:` are extremely low-entropy.
#   3. EXECUTION success stays LOW. A ~1M-parameter model should learn to write code that PARSES
#      long before it writes code that RUNS, and running requires the logic to be right.
#   4. Concepts pass better than 2/38, but the operator concepts (has_addition, has_comparison,
#      has_slicing) will be the hardest, because a single `+` token in a 16-token window is a weak
#      signal and these are common everywhere.
#   5. NOT a coding assistant. If execution success is 0/25, that is the honest result and it is
#      reported as 0/25.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the GPU..."
while pgrep -f "train_chat.py|train_code.py|run_best.sh" > /dev/null; do
  sleep 20
done
echo "GPU free; starting the code run"

CODE_MAX_PAIRS=60000 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs "${CODE_EPOCHS:-16}" --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --device gpu --out code1 --max-windows 1500000 --n-exec 25 2>&1 | tail -70

echo
echo "################################################################"
echo "# CODE — what runs, what does not"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/code"
for tag, out in [("code smoke", "codesmoke"), ("code real", "code1")]:
    f = os.path.join(P, f"{out}_replies.json")
    if not os.path.exists(f):
        print(f"  {tag}: -- not run --")
        continue
    r = json.load(open(f))
    h = r["history"][-1]
    e = r["exec"]
    print(f"  {tag}: ppl {h['val_ppl']:9.2f}  acc {h['val_acc']*100:5.2f}%  "
          f"syntax {e['syntax_ok']}/{e['total']}  RUNS {e['ran']}/{e['total']}  "
          f"concepts {r['audit']['n_pass']}/{r['audit']['n_concepts']}  params {r['params']:,}")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo "docs copied"
