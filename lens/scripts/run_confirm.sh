#!/bin/bash
# run_confirm.sh — confirm the win, finish the two broken arms.
#
# 1. 3 SEEDS of the wide arm. A single run is a strong lead, not a proven result, and the seed rule
#    is not waived for a good number.
# 2. The HYBRID arm, which never ran (a bug stopped it before it started).
# 3. The ROUND-TRIP with full output, to find why it crashed.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export BLOCKER_DEVICE=gpu
export BLOCKER_EPOCHS="${CF_EPOCHS:-40}"
export BLOCKER_LC=0.3
export BLOCKER_KIND=wide

echo "################################################################"
echo "# 1. THREE SEEDS OF THE WIDE ARM (the win must survive seed noise)"
echo "################################################################"
for s in 1 2; do
  echo "--- seed $s ---"
  BLOCKER_SEED=$s BLOCKER_OUT="opt1_wide60_seed$s" bash lens/scripts/run_blocker.sh 2>&1 \
    | grep -E "concept set|DONE|readable |normal |GAP|readable  " | tail -6
done

echo
echo "################################################################"
echo "# 2. THE HYBRID ARM (never ran before -- bug fixed)"
echo "################################################################"
BLOCKER_KIND=hybrid BLOCKER_OUT=opt3_hybrid bash lens/scripts/run_blocker.sh 2>&1 | tail -24

echo
echo "################################################################"
echo "# 3. THE ROUND-TRIP, full output"
echo "################################################################"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/roundtrip.py 2>&1 | tail -45

echo
echo "################################################################"
echo "# SEED SUMMARY"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs"
n = json.load(open(f"{P}/blocker/normal_summary.json"))["val_ppl"]
print(f"  matched normal ppl {n:.3f}   BAR ~6")
print(f"  {'arm':<26}{'concepts':>10}{'ppl':>10}{'gap':>10}{'acc':>9}")
gaps = []
for tag, d in [("wide seed 0", "opt1_wide60"), ("wide seed 1", "opt1_wide60_seed1"),
               ("wide seed 2", "opt1_wide60_seed2"), ("hybrid", "opt3_hybrid")]:
    f = f"{P}/{d}/readable_summary.json"
    if not os.path.exists(f):
        print(f"  {tag:<26}{'-- not run --':>10}")
        continue
    r = json.load(open(f))
    gap = r["val_ppl"] - n
    ct = f"{r['n_pass']}/{r['n_concepts']}"
    print(f"  {tag:<26}{ct:>10}{r['val_ppl']:>10.3f}{gap:>+10.3f}{r['val_acc']*100:>8.2f}%")
    if tag.startswith("wide"):
        gaps.append(gap)
if gaps:
    import statistics as st
    print(f"\n  wide arm across {len(gaps)} seeds: gap mean {st.mean(gaps):+.3f} "
          f"std {st.pstdev(gaps):.3f}   min {min(gaps):+.3f}  max {max(gaps):+.3f}")
    print(f"  {'WIN CONFIRMED' if max(gaps) < 6 else 'NOT CONFIRMED — some seed exceeds the bar'}")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md METER-06-GRADED.md FIX-01.md /home/yahya/ 2>/dev/null
echo "docs copied"
