#!/bin/bash
# run_scaling.sh — G2: does the win CARRY ACROSS SIZES?
#
# The owner asked directly: "if we move to bigger model and we see more improvement easily".
# That is exactly Target G2, which FAILED on the old engine and has never been re-tested on the
# rebuilt one. With the capacity threshold now known to matter, it is the right time.
#
# The test: the SAME wide concept set (41 concepts) at three model widths. If the win is real it
# should hold or improve as the model grows; if it is an artefact of one size, it will not.
#
# Pre-registered predictions (written before running):
#   1. The gap stays under the bar at every width, because the fix is the CONCEPT CAPACITY, which
#      does not change with width.
#   2. Bigger models get better on accuracy at every size for BOTH arms, readable and normal.
#   3. If the readable gap WIDENS with width, then the bottleneck is not only concept count and the
#      win is size-specific -- which must be reported as a failure to carry.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export BLOCKER_DEVICE=gpu
export BLOCKER_EPOCHS="${SC_EPOCHS:-40}"
export BLOCKER_LC=0.3
export BLOCKER_KIND=wide

for d in 128 512 1024; do
  echo
  echo "################################################################"
  echo "# WIDTH d=$d — readable and matched normal"
  echo "################################################################"
  BLOCKER_D=$d BLOCKER_OUT="scale_d$d" BLOCKER_ONLY=readable bash lens/scripts/run_blocker.sh 2>&1 \
    | grep -E "DONE|readable |GAP" | tail -4
  BLOCKER_D=$d BLOCKER_OUT="scale_d$d" BLOCKER_ONLY=normal BLOCKER_LC=0 bash lens/scripts/run_blocker.sh 2>&1 \
    | grep -E "DONE|normal |GAP" | tail -4
done

echo
echo "################################################################"
echo "# SCALING SUMMARY"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs"
print(f"  {'width':<8}{'arm':<10}{'params':>11}{'concepts':>10}{'ppl':>10}{'acc':>9}{'gap':>9}")
res = {}
for d in (128, 256, 512, 1024):
    dirn = f"scale_d{d}" if d != 256 else "opt1_wide60"
    rd = f"{P}/{dirn}/readable_summary.json"
    nd = f"{P}/{dirn}/normal_summary.json"
    if not os.path.exists(rd):
        print(f"  {d:<8}{'-- not run --':>10}")
        continue
    r = json.load(open(rd))
    ct = f"{r['n_pass']}/{r['n_concepts']}"
    if os.path.exists(nd):
        n = json.load(open(nd))
        gap = r["val_ppl"] - n["val_ppl"]
        print(f"  {d:<8}{'readable':<10}{r['params']:>11,}{ct:>10}{r['val_ppl']:>10.3f}"
              f"{r['val_acc']*100:>8.2f}%{gap:>+9.3f}")
        print(f"  {'':<8}{'normal':<10}{n['params']:>11,}{'-':>10}{n['val_ppl']:>10.3f}"
              f"{n['val_acc']*100:>8.2f}%{'':>9}")
        res[d] = (gap, r["val_acc"], n["val_acc"])
    else:
        print(f"  {d:<8}{'readable':<10}{r['params']:>11,}{ct:>10}{r['val_ppl']:>10.3f}"
              f"{r['val_acc']*100:>8.2f}%{'n/a':>9}")
        res[d] = (None, r["val_acc"], None)
print()
gaps = [(d, g) for d, (g, _, _) in res.items() if g is not None]
if gaps:
    for d, g in sorted(gaps):
        print(f"  d={d:<6} gap {g:+.3f}   {'UNDER the bar (~6)' if g < 6 else 'OVER the bar'}")
    if all(g < 6 for _, g in gaps):
        print("\n  WIN CARRIES ACROSS SIZES -- every width is under the bar.")
    else:
        print("\n  WIN DOES NOT CARRY -- at least one width is over the bar.")
    if len(gaps) >= 2:
        gs = [g for _, g in sorted(gaps)]
        trend = "widens" if gs[-1] > gs[0] else "narrows or holds"
        print(f"  trend with width: the gap {trend} "
              f"({gs[0]:+.3f} at the smallest -> {gs[-1]:+.3f} at the largest)")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md METER-06-GRADED.md FIX-01.md /home/yahya/ 2>/dev/null
echo "docs copied"
