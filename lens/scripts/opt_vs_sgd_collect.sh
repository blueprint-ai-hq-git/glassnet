#!/bin/bash
# opt_vs_sgd_collect.sh -- print the comparison TABLE for PLAN item (d2b) from the arms' own append-only
# per-epoch rows. Appends to lens/runs/OPT-VS-SGD-TABLE.txt (never overwrites). Runs as the LAST queue
# entry, after every arm, so the table is built from complete runs.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
OUT=lens/runs/OPT-VS-SGD-TABLE.txt
{
  echo "=== OPT-VS-SGD TABLE $(date '+%F %H:%M:%S')"
  echo "    built from lens/runs/chat/opt_<arm>_s<seed>_history.jsonl (one row per epoch per arm)"
  .venv-gpu/bin/python3 -u lens/scripts/opt_vs_sgd_table.py
  echo "COLLECT EXIT=$?"
  echo "=== TABLE DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
