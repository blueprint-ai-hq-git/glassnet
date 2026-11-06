#!/bin/bash
# Audit the K21h seed runs once the trainer exits. CPU only (the GPU is handed to the
# distillation run), so this waits for train_seetalk to disappear, then audits whichever
# seed checkpoints exist and records the verdicts to lens/runs/seetalk/k21h_audit.txt.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
OUT=lens/runs/seetalk/k21h_audit.txt
echo "=== K21h audits $(date '+%F %H:%M') ===" > "$OUT"
# wait loop removed: pgrep -f matched the agent wrapper processes (phantom match);
# the trainer is already gone, so the audits start immediately
for S in 1 2; do
  F="lens/runs/seetalk/seetalk_k21h_s${S}.npz"
  if [ -f "$F" ]; then
    echo "--- seed $S ---" >> "$OUT"
    VISION_SHIP="seetalk_k21h_s${S}.npz" nice -n 19 ionice -c3 \
      .venv-gpu/bin/python3 -u lens/scripts/audit_strict_vision.py >> "$OUT" 2>&1
    echo "exit=$?" >> "$OUT"
  else
    echo "--- seed $S: NO CHECKPOINT (run failed) ---" >> "$OUT"
  fi
done
echo "=== done $(date '+%F %H:%M') ===" >> "$OUT"