#!/bin/bash
# run_threshold.sh — where does the capacity threshold sit?
#
# 22 concepts -> gap +20.203 (FAILED)
# 41 concepts -> gap +0.372  (WIN, confirmed on 3 seeds)
#
# The relationship is NOT "more is better" -- there is a threshold. Bisecting it is now a bounded
# question with a pre-registered concept set (truncated, never re-chosen).
#
# Pre-registered predictions (written before running):
#   The threshold should be where the answer layer has enough independent numbers to reconstruct
#   the local context. I expect it between 28 and 34. If 28 already wins, the threshold is lower
#   than the hybrid suggested and the hybrid's failure was about WHICH concepts (the extra 13 had
#   very low support), not how many.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export BLOCKER_DEVICE=gpu
export BLOCKER_EPOCHS="${TH_EPOCHS:-40}"
export BLOCKER_LC=0.3
export BLOCKER_KIND=wide

echo "=== recording the confirmed win ==="
.venv-gpu/bin/python3 -u lens/scripts/record_confirmed.py 2>&1 | tail -2

for k in 28 34; do
  echo
  echo "################################################################"
  echo "# THRESHOLD BISECT: $k concepts"
  echo "################################################################"
  BLOCKER_K=$k BLOCKER_OUT="th_k$k" bash lens/scripts/run_blocker.sh 2>&1 \
    | grep -E "capped|DONE|readable |normal |GAP|readable   " | tail -6
done

echo
echo "################################################################"
echo "# THE ROUND-TRIP (fixed)"
echo "################################################################"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/roundtrip.py 2>&1 | tail -40

echo
echo "################################################################"
echo "# THRESHOLD SUMMARY"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs"
n = json.load(open(f"{P}/blocker/normal_summary.json"))["val_ppl"]
print(f"  matched normal ppl {n:.3f}    BAR ~6")
print(f"  {'concepts':<12}{'passed':>9}{'ppl':>10}{'gap':>10}{'acc':>9}   verdict")
rows = [("9", "fix01_lc0.3"), ("22", "opt3_hybrid"), ("28", "th_k28"), ("34", "th_k34"),
        ("41", "opt1_wide60")]
for tag, d in rows:
    f = f"{P}/{d}/readable_summary.json"
    if not os.path.exists(f):
        print(f"  {tag:<12}{'-- not run --':>9}")
        continue
    r = json.load(open(f))
    gap = r["val_ppl"] - n
    ct = f"{r['n_pass']}/{r['n_concepts']}"
    v = "WIN" if gap < 6 else "FAIL"
    print(f"  {tag:<12}{ct:>9}{r['val_ppl']:>10.3f}{gap:>+10.3f}{r['val_acc']*100:>8.2f}%   {v}")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md METER-06-GRADED.md FIX-01.md /home/yahya/ 2>/dev/null
echo "docs copied"
