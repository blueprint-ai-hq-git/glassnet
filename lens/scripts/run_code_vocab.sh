#!/bin/bash
# run_code_vocab.sh — CHANGE 2: the vocabulary, at 32k instead of 6k.
#
# THE MEASURED NEED (measure_vocab_need.py, code-only corpus):
#     vocab    <unk> rate
#      6,000      7.18%
#     12,000      5.58%
#     20,000      4.52%
#     32,000      3.51%
#     50,000      2.47%
#   on a REAL 322-token solution: <unk> at 6k = 26 tokens (8.1%), at 32k = 1 token (0.3%)
#
# WHERE WE ARE (fair, budget-matched):
#   code1        both corpus  16 ep  ppl 44.26  acc 36.44%  syntax 4/25
#   codedoc_full code corpus  16 ep  ppl 39.05  acc 38.21%  syntax 6/25
# The corpus change is confirmed: better ppl, better acc, better executability. And `<unk>` is still
# the most common token in the generated output, which the corpus change could not fix.
#
# THIS RUN changes ONE thing: CD_VOCAB 6000 -> 32000. Everything else identical to codedoc_full.
#
# Pre-registered predictions:
#   1. Generated text contains far fewer <unk>, and identifiers become expressible -- `def <unk> (`
#      should start becoming `def <name> (`.
#   2. Syntax rate rises WELL above 6/25, because most syntax errors come from <unk> sitting where a
#      name must be.
#   3. ppl gets WORSE or flat: a bigger vocabulary means more tokens to predict, so per-token
#      perplexity rises even when the text gets better. If ppl IMPROVES I should be suspicious.
#   4. Model size grows ~4x (the embedding is the dominant term) -- reported, and it must fit the
#      project's limits.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CODE_DOC=code
export CD_VOCAB=32000

echo "############ CHANGE 2: vocab 6k -> 32k, corpus code-only, budget matched ############"
CD_VOCAB=32000 CODE_DOC=code .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs "${CE:-16}" --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
  --device gpu --max-windows 1200000 --n-exec 25 --out covocab 2>&1 | tail -26

echo
echo "############ the executable test ############"
CKPT=covocab_ep${CE:-16} CD_VOCAB=32000 CODE_DOC=code .venv-gpu/bin/python3 -u \
  lens/scripts/reexec_code.py 2>&1 | tail -8

echo
echo "############ CODE: every step, budget-matched ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/code"
print(f"  {'run':<20}{'corpus':<8}{'vocab':>8}{'ppl':>9}{'acc':>9}{'params':>11}")
for name, corpus, vocab in (("code1", "both", 6000), ("codedoc_full", "code", 6000),
                            ("covocab", "code", 32000)):
    f = os.path.join(P, f"{name}_log.json")
    if not os.path.exists(f):
        print(f"  {name:<20}{corpus:<8}{vocab:>8,}   not run")
        continue
    d = json.load(open(f)); h = d["history"][-1]
    print(f"  {name:<20}{corpus:<8}{vocab:>8,}{h['val_ppl']:>9.2f}{h['val_acc']*100:>8.2f}%"
          f"{d.get('params', 0):>11,}")
rx = os.path.join(P, "code1_reexec.json")
if os.path.exists(rx):
    s = json.load(open(rx))["stats"]
    print(f"\n  executable test (latest): syntax {s['syntax']}/{s['total']}, "
          f"runs {s['ran']}/{s['total']}")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
