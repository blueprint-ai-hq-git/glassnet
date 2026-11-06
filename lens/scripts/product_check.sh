#!/bin/bash
# PRODUCT CHECK - the one command a normal person runs.
# It (1) checks every shipped line against its own gates, (2) refreshes the pictures, (3) prints a
# plain verdict. Nobody has to know about corpora, window sizes or eval meters to use this.
# Anything that fails is printed as FAILED with the line it came from; nothing is skipped silently.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
echo "=============================================================================="
echo " GlassNet product check   $(date '+%Y-%m-%d %H:%M')"
echo "=============================================================================="
echo
echo "1. checking every line against its own gates (this is the reliability part)"
echo "------------------------------------------------------------------------------"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/selfcheck.py > /tmp/product_check_selfcheck.log 2>&1
SC=$?
grep -E "^  \[PASS\]|^  \[FAIL\]|SELFCHECK:" /tmp/product_check_selfcheck.log | tail -60
if [ "$SC" -eq 0 ]; then
  echo
  echo "   RESULT: every check passed."
else
  echo
  echo "   RESULT: FAILED (exit $SC). The failing lines are the [FAIL] rows above."
  echo "   Full output: /tmp/product_check_selfcheck.log"
fi
echo
echo "2. refreshing the pictures"
echo "------------------------------------------------------------------------------"
/home/yahya/hermes-env/bin/python3 -u lens/figures.py 2>&1 | grep -E "figure:|manifest|could not"
echo
echo "3. what the pictures say right now"
echo "------------------------------------------------------------------------------"
/home/yahya/hermes-env/bin/python3 -u - <<'PY'
import json
r = json.load(open("lens/runs/HONEST-LADDER.json"))["rows"]
best = max(r, key=lambda x: x["honest"])
print(f"   best next-word accuracy on truly unseen dialogue: {best['honest']:.2f}%  ({best['ship']})")
print(f"   models compared: {len(r)}   all of them between "
      f"{min(x['honest'] for x in r):.2f}% and {max(x['honest'] for x in r):.2f}%")
l = json.load(open("lens/runs/LINES.json"))
print(f"   lines whose own claims were verified true: {len(l['label_honesty'])} of "
      f"{len(l['label_honesty'])}")
print(f"   capability numbers we have NOT measured yet: "
      f"{', '.join(l['capability_not_measured'])}")
PY
echo
echo "=============================================================================="
echo " pictures: lens/runs/figures/  (how_good_really, does_it_talk, what_works, glassnet_picture)"
echo " results ledger: lens/runs/BENCH.md   bugs: BUGS.md   run log: LOG.md"
echo "=============================================================================="