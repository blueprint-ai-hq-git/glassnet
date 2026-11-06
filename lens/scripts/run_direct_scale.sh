#!/bin/bash
# run_direct_scale.sh (2026-10-22 21:2x self-wake). THE P75 LEAD TESTED AS A LEVER.
# PREDICTIONS P77a-g: lens/runs/DIRECT-SCALE-PREDICTIONS-P77.txt — written BEFORE the instrument existed.
# Scales the TIED DIRECT term of the readout by 1.0 / 0.5 / 0.25 / 0.0 at inference on the two best
# checkpoints and measures the text, the hedge count, the <unk> share, the prompt-steering share of the
# first-token top-20 (REAL vs MISMATCH) and the rank of the TRUE reply token.
# WRITES ONLY: lens/runs/DIRECT-SCALE.txt + lens/runs/DIRECT-SCALE-RUN.txt.
# CPU only, 2 BLAS threads, nice 19, no GPU use, never trains, no checkpoint is written.
# USAGE: run_direct_scale.sh [smoke]   (smoke = 2 prompts, 2 scales, output into lens/runs/_tmp/, 10s)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
MODE="${1:-full}"
if [ "$MODE" = "smoke" ]; then
  export SCALE_SMOKE=1
  export SCALE_OUT="$PWD/lens/runs/_tmp/DIRECT-SCALE-SMOKE.txt"
  LOGF="$PWD/lens/runs/_tmp/DIRECT-SCALE-SMOKE-RUN.txt"
else
  export SCALE_OUT="$PWD/lens/runs/DIRECT-SCALE.txt"
  LOGF="$PWD/lens/runs/DIRECT-SCALE-RUN.txt"
fi
{
  echo "=== direct-scale runner ($MODE) $(date '+%F %H:%M:%S') ==="
  echo "predictions: lens/runs/DIRECT-SCALE-PREDICTIONS-P77.txt (P77a-g)"
  echo "checkpoints read (never written):"
  for f in lens/runs/chat/ct_d256_full169_fix3_best.npz lens/runs/chat/ct_d512_full169_fix3_best.npz; do
    echo "  $(stat -c '%y %s bytes %n' "$f")"
  done
  start=$(date +%s)
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/direct_scale_test.py
  rc=$?
  echo "python exit=$rc   wall=$(( $(date +%s) - start ))s"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> "$LOGF" 2>&1
