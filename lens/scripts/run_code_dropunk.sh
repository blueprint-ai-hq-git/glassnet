#!/bin/bash
# run_code_dropunk.sh — CHANGE 3: stop rewarding `<unk>`.
#
# MEASURED, on the code-only corpus at 6k vocab:
#     <unk> is 6.98% of all TRAINING TARGETS (62,937 of 902,310) -- the #1 most frequent token the
#     model is asked to predict. At 32k vocab it is 3.48% and drops out of the top five.
# So a plain next-token objective spends real capacity learning to EMIT `<unk>`. That is why
# enlarging the vocabulary 5x (ppl 39.05 -> 54.30, params 1.5M -> 7.5M) did NOT improve executability:
# a bigger vocabulary changes how often the INPUT is unknown; it does not change what the model is
# REWARDED for producing.
#
# The positions whose target is `<unk>` are unanswerable BY CONSTRUCTION -- the true token was
# discarded by the vocabulary before training began. Training on them can only teach `<unk>`.
#
# THE ALTERNATIVE I DID NOT TAKE: `ignore_index` inside the loss, which is the standard framework
# approach. `softmax_ce` exists in THREE implementations here (numpy, the C library `liblens.so`, and
# the CUDA path), so that would be three changes, each needing its own gradient check. Dropping the
# windows at the data layer is ONE change, loses no context (windows overlap at stride W/2), and is
# trivially inspectable. Recorded so the choice is visible rather than implicit.
#
# ONE CHANGE from codedoc_full: + --drop-unk. Same corpus, same 6k vocab, same budget, same seed.
#
# Pre-registered predictions:
#   1. The generated `<unk>` rate falls sharply. It cannot go to zero -- `<unk>` can still be
#      produced from an unknown INPUT token.
#   2. Syntax/executable rate rises above 6/25 on ALL prompts, and above 2/5 on the clean ones.
#   3. val ppl gets WORSE. The model is no longer allowed to score easy points on the 7% of targets
#      that were `<unk>`, so the remaining targets are harder. A ppl improvement here would be
#      suspicious.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CODE_DOC=code
export CD_VOCAB=6000

echo "############ CHANGE 3: --drop-unk (stop rewarding <unk>), budget matched to codedoc_full ############"
CODE_DOC=code CD_VOCAB=6000 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs "${CE:-16}" --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
  --device gpu --max-windows 1200000 --n-exec 25 --drop-unk --out codropunk 2>&1 | tail -30

echo
echo "############ the executable test, both prompt sets ############"
CKPT=codropunk_ep${CE:-16} CD_VOCAB=6000 CODE_DOC=code .venv-gpu/bin/python3 -u \
  lens/scripts/reexec_code.py 2>&1 | tail -12

echo
echo "############ CODE: the whole ladder, each step budget-matched ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/code"
print(f"  {'run':<18}{'corpus':<7}{'vocab':>7}{'unk tgt':>9}{'ppl':>8}{'acc':>8}{'params':>11}")
for name, corpus, vocab, unk in (("code1", "both", 6000, "6.98%"),
                                 ("codedoc_full", "code", 6000, "6.98%"),
                                 ("covocab", "code", 32000, "3.48%"),
                                 ("codropunk", "code", 6000, "MASKED")):
    f = os.path.join(P, f"{name}_log.json")
    if not os.path.exists(f):
        print(f"  {name:<18}{corpus:<7}{vocab:>7,}{unk:>9}   not run")
        continue
    d = json.load(open(f)); h = d["history"][-1]
    print(f"  {name:<18}{corpus:<7}{vocab:>7,}{unk:>9}{h['val_ppl']:>8.2f}{h['val_acc']*100:>7.2f}%"
          f"{d.get('params', 0):>11,}")
print()
for tag, f in (("all prompts", "code1_reexec.json"), ("clean prompts", "code1_reexec_clean.json")):
    p = os.path.join(P, f)
    if os.path.exists(p):
        s = json.load(open(p))["stats"]
        print(f"  {tag:<14} syntax {s['syntax']}/{s['total']}  runs {s['ran']}/{s['total']}"
              f"   ({s['ran']/max(s['total'],1)*100:.0f}%)")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
