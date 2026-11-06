#!/bin/bash
# run_options.sh — all of the owner's options, plus the token round-trip.
#
# Owner: "just try all of the options ... and did you do the token thing so like each token means
# something and the number exactly and not cheating ... it should be fully transparent and tokens
# convertible ... it can output multiple number ... it has one neural can get more data then it was
# able to do before".
#
# OPTION 1  wide60  — 60 named concepts instead of 9. More numbers for the answer layer to read.
# OPTION 2  ceiling — no run; the honest report of what the trade-off costs.
# OPTION 3  hybrid  — 9 audited concepts PLUS extra named concepts, so explain() stays short while
#                     the model is not starved.
# OPTION 4  roundtrip — model -> words -> model, the "tokens convertible" test.
#
# Pre-registered predictions (written before running):
#   wide60  : the gap closes toward the bar, because 60 numbers carry more than 9. The audit may
#             drop, since more concepts means more that can fail.
#   hybrid  : the gap closes AND the 9 audited concepts stay at 9/9. This is the only option that
#             could beat the blocker without giving up readability.
#   roundtrip: every printed token weight equals the model's own weight exactly (to float
#             precision), and the residual column shows honestly what the top-N list misses.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export BLOCKER_DEVICE=gpu
export BLOCKER_EPOCHS="${OPT_EPOCHS:-40}"
export BLOCKER_LC=0.3

echo "################################################################"
echo "# OPTION 4 FIRST (fast, and it is the direct question): ROUND-TRIP"
echo "################################################################"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/roundtrip.py 2>&1 | tail -40

echo
echo "################################################################"
echo "# OPTION 1: WIDE — 60 named concepts instead of 9"
echo "################################################################"
BLOCKER_KIND=wide BLOCKER_OUT=opt1_wide60 bash lens/scripts/run_blocker.sh 2>&1 | tail -26

echo
echo "################################################################"
echo "# OPTION 3: HYBRID — 9 audited + extra named concepts"
echo "################################################################"
BLOCKER_KIND=hybrid BLOCKER_OUT=opt3_hybrid bash lens/scripts/run_blocker.sh 2>&1 | tail -26

echo
echo "################################################################"
echo "# OPTION 2: THE CEILING — every arm, side by side"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs"
n = json.load(open(f"{P}/blocker/normal_summary.json"))["val_ppl"]
print(f"  matched normal ppl {n:.3f}    BAR ~6")
print(f"  {'arm':<24}{'concepts':>10}{'ppl':>10}{'gap':>10}{'acc':>9}")
rows = [("lc=0.1 binary", "blocker"), ("lc=0.3 binary", "fix01_lc0.3"),
        ("lc=0.5 binary", "fix01_lc0.5"), ("lc=1.0 binary", "fix01_lc1"),
        ("two-phase", "twophase"), ("FIX-02 soft MSE", "fix02_soft_lc0.3"),
        ("FIX-03 soft BCE", "fix03_softbce_lc0.3"),
        ("OPT1 wide 60 concepts", "opt1_wide60"), ("OPT3 hybrid", "opt3_hybrid")]
best = None
for tag, d in rows:
    f = f"{P}/{d}/readable_summary.json"
    if not os.path.exists(f):
        f = f"{P}/{d}/summary.json"
    if not os.path.exists(f):
        print(f"  {tag:<24}{'-- not run --':>10}")
        continue
    r = json.load(open(f))
    ppl = r["val_ppl"]; gap = ppl - n
    passed, total = r["n_pass"], r["n_concepts"]
    print(f"  {tag:<24}{f'{passed}/{total}':>10}{ppl:>10.3f}{gap:>+10.3f}{r['val_acc']*100:>8.2f}%")
    if best is None or gap < best[1]:
        best = (tag, gap)
if best:
    print(f"\n  BEST ARM: {best[0]} at {best[1]:+.3f}   (bar ~6)")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md METER-06-GRADED.md FIX-01.md /home/yahya/ 2>/dev/null
echo "docs copied"
