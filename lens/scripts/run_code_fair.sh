#!/bin/bash
# run_code_fair.sh — the code-only change at the SAME budget as code1, because the last comparison
# was unfair and I reported its number anyway.
#
# WHAT WENT WRONG
#   codedoc  (code-only)  4 epochs  120,000 windows   ppl 91.06  acc 28.97%  -> 2/25 syntax
#   code1    (both)      16 epochs 1,200,000 windows  ppl 44.26  acc 36.44%  -> 4/25 syntax
#
# 40x less training for codedoc. Reporting "2/25, worse than 4/25" from that pair compares training
# BUDGET, not the corpus change. This is the same failure as the big-ladder bug: two runs of
# different sizes presented as a comparison.
#
# The pre-registered prediction from the corpus measurement was: "generated text stops being prose;
# syntax improves from 4/25 but stays LOW, because <unk> is still the most common token." That
# prediction is NOT yet tested, because the run that was supposed to test it was under-budget.
#
# THIS RUN: identical to code1 -- d=192, window=16, epochs=16, batch=256, lr=0.4, lr-decay=0.5,
# lam-concept=0.3, max-windows=1200000, device=gpu, seed=0. Only CODE_DOC differs.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CODE_DOC=code

echo "############ code-only corpus, budget MATCHED to code1 (1.2M windows, 16 epochs) ############"
CODE_DOC=code .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs "${CE:-16}" --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
  --device gpu --max-windows 1200000 --n-exec 25 --out codedoc_full 2>&1 | tail -30

echo
echo "############ the executable test, budget-matched ############"
CKPT=codedoc_full_ep${CE:-16} CODE_DOC=code .venv-gpu/bin/python3 -u \
  lens/scripts/reexec_code.py 2>&1 | tail -8

echo
echo "############ the fair comparison ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/code"
print(f"  {'run':<22}{'corpus':<16}{'epochs':>8}{'windows':>10}{'ppl':>9}{'acc':>8}{'syntax':>9}{'runs':>7}")
rows = [("code1", "both", 16, 1200000), ("codedoc (UNFAIR)", "code", 4, 120000),
        ("codedoc_full", "code", 16, 1200000)]
for name, corpus, ep, w in rows:
    f = os.path.join(P, f"{name.split()[0]}_log.json")
    if name == "codedoc_full":
        f = os.path.join(P, "codedoc_full_log.json")
    if not os.path.exists(f):
        print(f"  {name:<22}{corpus:<16}{ep:>8}{w:>10,}   not run")
        continue
    d = json.load(open(f)); h = d["history"][-1]
    print(f"  {name:<22}{corpus:<16}{ep:>8}{w:>10,}{h['val_ppl']:>9.2f}{h['val_acc']*100:>7.2f}%"
          f"{'':>9}{'':>7}")
rx = os.path.join(P, "code1_reexec.json")
if os.path.exists(rx):
    print(f"\n  executable test after the LATEST reexec: {json.load(open(rx))['stats']}")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
