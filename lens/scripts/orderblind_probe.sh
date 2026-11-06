#!/bin/bash
# orderblind_probe.sh -- WIRING PROBE for lens/scripts/orderblind_test.py (P117f determinism + the loader/mask
# path) on 256 windows, ONE checkpoint, in the scratch dir so the real record file stays clean.
# This is a WIRING check only: P117's bands are NOT decided here and no number from it is quoted as a result.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export EVAL_DEVICE=cpu
export PYTHONUNBUFFERED=1
export OB_MAXWIN=256
export CKPT_LIST="TWIN probe=lens/runs/chat/ct_d256_full169_fix3_ep1.npz"
OUT="/home/yahya/.hermes/cache/scratch/orderblind_probe.txt"
{
  echo "# WIRING PROBE $(date '+%F %H:%M:%S') -- 256 windows, ep1 checkpoint, NOT the record run"
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/orderblind_test.py
  echo "# probe exit=$?"
} > "$OUT" 2>&1
tail -45 "$OUT"
