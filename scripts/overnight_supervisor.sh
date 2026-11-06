#!/bin/bash
# overnight_supervisor.sh — unattended runner for tonight.
#   1. Keeps STEP 2 (5 seeds x 2 models) alive until 10/10 runs are banked:
#      if the external killer takes the sweep, this restarts it (the sweep
#      itself skips already-completed runs, so nothing is re-trained).
#   2. Then runs STEP 4 (majority baseline is already in Step 1; this is the
#      capacity + longer-window probe).
# Safe to relaunch at any time — every stage is idempotent.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t2/step2 runs/t2/step4
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T2_MAX_EPOCHS=96
export T2_PATIENCE=15
PY=/home/yahya/.hermes/hermes-agent/venv/bin/python3
SUP=runs/t2/step2/supervisor.log

count_runs() {
  "$PY" -c 'import json,os
p="runs/t2/step2/sweep.json"
d=json.load(open(p)) if os.path.exists(p) else []
runs=d.get("runs",d) if isinstance(d,dict) else d
print(len(runs))' 2>/dev/null || echo 0
}

echo "$(date '+%F %T') supervisor: start (banked $(count_runs)/10)" >> "$SUP"

while true; do
  n=$(count_runs)
  if [ "$n" -ge 10 ]; then
    echo "$(date '+%F %T') supervisor: step2 complete ($n/10)" >> "$SUP"
    break
  fi
  if pgrep -f "[t]2_step2_sweep" > /dev/null; then
    sleep 60
  else
    echo "$(date '+%F %T') supervisor: sweep absent ($n/10 banked) -> resuming" >> "$SUP"
    nice -n 19 ionice -c3 "$PY" scripts/t2_step2_sweep.py \
      >> runs/t2/step2/sweep.log 2>&1
  fi
done

if [ ! -f runs/t2/step4/results.json ]; then
  echo "$(date '+%F %T') supervisor: starting STEP 4" >> "$SUP"
  T2_MAX_EPOCHS=32 T2_PATIENCE=8 nice -n 19 ionice -c3 \
    "$PY" scripts/t2_step4_ceiling.py >> runs/t2/step4/step4.log 2>&1
  echo "$(date '+%F %T') supervisor: STEP 4 exit=$?" >> "$SUP"
fi
echo "$(date '+%F %T') supervisor: ALL DONE" >> "$SUP"