#!/bin/bash
# append_p115e_recheck.sh -- append the corrected P115e check output to the append-only log, with one line
# saying which run it corrects. The FIRST run of the checker printed "FAIL" from a parser that mixed the
# stored file's two checkpoint blocks; that first output stays in the log as the record (below), unchanged.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
OUT=lens/runs/SENTSTATE-DECODE.txt
.venv-gpu/bin/python3 lens/scripts/sentstate_decider_control_check.py > lens/runs/_tmp/p115e_check.txt 2>&1
echo "RECHECK EXIT=$?" >> lens/runs/_tmp/p115e_check.txt
{
  echo ""
  echo "=== P115e RE-CHECK after the CHECKER's own defect was fixed (BUGS.md: the first checker mixed the"
  echo "=== stored file's ARM block and its TWIN block and so printed 'GREEDY TOKENS 0/6 identical' beside six"
  echo "=== strings that were in fact identical; the block-1-vs-block-1 comparison below is the same measurement)"
  cat lens/runs/_tmp/p115e_check.txt
} >> "$OUT"
tail -4 "$OUT"
