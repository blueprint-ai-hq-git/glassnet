#!/bin/bash
# CPU-only val-split audit (predictions P106a-c inside lens/scripts/val_split_by_source.py, written before its
# first run). Light: sums tokens per conversation from the mem-mapped bounds/ids arrays; no model, no GPU, no
# trainer corpus loader. Cores 2-3, nice 19 (the training arm is pinned to 0,1). Output: a NEW file
# lens/runs/VAL-SPLIT-BY-SOURCE.txt — nothing overwritten.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
{
  echo "======================================================================================================"
  echo "VAL SPLIT BY SOURCE GROUP (CPU only, cores 2-3, nice 19)  $(date '+%F %H:%M:%S')"
  echo "predictions P106a-c are inside the script's own docstring, written BEFORE this runner existed"
  echo "======================================================================================================"
  taskset -c 2,3 nice -n 19 ionice -c3 python3 -u lens/scripts/val_split_by_source.py
  echo "VALSPLIT EXIT=$?  $(date '+%F %H:%M:%S')"
} >> lens/runs/VAL-SPLIT-BY-SOURCE.txt 2>&1
tail -n 4 lens/runs/VAL-SPLIT-BY-SOURCE.txt
