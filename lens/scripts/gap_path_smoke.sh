#!/bin/bash
# gap_path_smoke.sh -- after the (d2d) edit, the GAP gate's own path must still run: the queued
# lens/scripts/run_gate_arm.sh uses --gates gap, and this turn restructured the block that builds it.
# Not a new bar: it re-runs the STORED gate_smoke.sh arm C (--gates gap --gate-variant off) and the shipped
# gap default, and checks the control still refuses 0.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_gate_small"
export PYTHONPATH="$PWD/lens/scripts"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/_tmp/gap_path_smoke.txt
TINY="--d 128 --window 32 --epochs 3 --batch 256 --lr 0.4 --lam-concept 0.3 --seed 0 --device cpu"
{
  echo "=== GAP PATH SMOKE $(date '+%F %H:%M:%S')"
  echo "--- --gates gap --gate-variant off (the CONTROL: must refuse 0 and match the ungated numbers)"
  nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $TINY --gates gap --gate-variant off \
    --probe-windows 256 --out _gap_path_ctl | tee lens/runs/_tmp/gap_path_ctl.txt
  echo "CTL EXIT=${PIPESTATUS[0]}"
  echo "--- --gates gap (the SHIPPED default, ema_band beta 0.5 tol 0.0: must run and log)"
  nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $TINY --gates gap \
    --probe-windows 256 --out _gap_path_def
  echo "DEF EXIT=$?"
  echo "--- --gates gap+agreement (both rules together: must run and write BOTH logs)"
  nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $TINY --gates gap+agreement \
    --probe-windows 256 --out _gap_path_both
  echo "BOTH EXIT=$?"
  echo "--- numbers-only diff of the GAP control against the STORED gate_smoke.sh arm C (the same command)"
  if diff <(grep -E '^  ep [0-9]+/' lens/runs/_tmp/gate_smoke_C.txt | sed 's/  ([0-9]*s)$//') \
          <(grep -E '^  ep [0-9]+/' lens/runs/_tmp/gap_path_ctl.txt 2>/dev/null | sed 's/  ([0-9]*s)$//'); then
    echo "gap control vs stored arm C: IDENTICAL"
  else
    echo "gap control vs stored arm C: DIFFERENT or no stored arm C to diff against"
  fi
  echo "=== GAP PATH SMOKE DONE $(date '+%F %H:%M:%S')"
} > "$OUT" 2>&1
