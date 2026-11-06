#!/bin/bash
# run_chain.sh — the rest of the owner's list, in order, one job after another.
#
# 1. SCALING (his direct question): does the win carry to bigger models?
# 2. DISTILLATION (G3): can a model make a smaller model smarter, same size, teacher precondition
#    enforced?
# 3. BENCHMARKS: everything this model can actually be measured on.
#
# Everything is sequential because there is one GPU. Each arm writes to its own directory; nothing
# is overwritten.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "################################################################"
echo "# 1. SCALING — does the win carry across model widths?"
echo "################################################################"
SC_EPOCHS=40 bash lens/scripts/run_scaling.sh 2>&1 | tail -30

echo
echo "################################################################"
echo "# 2. DISTILLATION — teacher -> small student vs same-size control"
echo "################################################################"
DS_EPOCHS=40 DS_DEVICE=gpu nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/run_distill.py 2>&1 | tail -40

echo
echo "################################################################"
echo "# 3. BENCHMARKS — everything this model can actually be measured on"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs"
print("  WHAT THIS MODEL CAN AND CANNOT BE BENCHMARKED ON")
print("  ---------------------------------------------------------------")
print("  CAN:     next-token perplexity and accuracy (its actual task)")
print("           the concept fidelity audit (does each name mean what it says)")
print("           a CHECKABLE task, where a wrong answer is provably wrong")
print("  CANNOT:  coding, general knowledge, reasoning benchmarks")
print("           it is a ~500k-parameter word-level model of Elizabethan English.")
print("           Scoring it on those would produce a number with no meaning.")
print()
print("  MEASURED SO FAR")
print("  ---------------------------------------------------------------")
n = json.load(open(f"{P}/blocker/normal_summary.json"))["val_ppl"]
print(f"  matched normal ppl {n:.3f}    BAR on the gap ~6")
print(f"  {'arm':<26}{'concepts':>10}{'ppl':>10}{'gap':>10}{'acc':>9}")
for tag, d in [("lc=0.1 (9 concepts)", "blocker"), ("two-phase (9, frozen)", "twophase"),
               ("FIX-02 soft MSE", "fix02_soft_lc0.3"), ("FIX-03 soft BCE", "fix03_softbce_lc0.3"),
               ("22 concepts (hybrid)", "opt3_hybrid"), ("28 concepts", "th_k28"),
               ("34 concepts", "th_k34"), ("41 concepts (WIN)", "opt1_wide60"),
               ("41 seed 1", "opt1_wide60_seed1"), ("41 seed 2", "opt1_wide60_seed2")]:
    f = f"{P}/{d}/readable_summary.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    gap = r["val_ppl"] - n
    ct = f"{r['n_pass']}/{r['n_concepts']}"
    print(f"  {tag:<26}{ct:>10}{r['val_ppl']:>10.3f}{gap:>+10.3f}{r['val_acc']*100:>8.2f}%")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md METER-06-GRADED.md FIX-01.md /home/yahya/ 2>/dev/null
echo "docs copied"
