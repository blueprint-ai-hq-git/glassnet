#!/bin/bash
# K21i — the LAST untried image lever: epochs 14 -> 20 (ONE change), same 19-set, lam 0.6,
# d128, seed 0, datum untouched. Rationale: the K21b->K21c history showed concept margins
# shifting with training length, so more epochs is the only remaining honest knob (capacity
# d160 was measured equal; seeds 1/2 measured equal; pruning measured wrong).
# BAR: strict 19/19 AND gates 4/4 -> ships. Else image closes as the measured maximum 17/19.
# Waits on the distillation run's PID (kill -0, NOT pgrep -f: that matched the agent's own
# wrapper processes and stalled two jobs earlier today).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CONCEPT_EXCLUDE="centre_heavier,has_fine_texture,left_right_symmetric,bottom_left_mass,bottom_right_mass,top_right_mass,centre_column,centre_row,edge_mass_bottom,mostly_empty,com_high,com_low,has_high_contrast,is_thin_structure"
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
DISTILL_PID=$(pgrep -f "run_distill_real.py" | head -1)
if [ -n "$DISTILL_PID" ]; then
  echo "waiting for the distillation run (pid $DISTILL_PID) to free the GPU..."
  while kill -0 "$DISTILL_PID" 2>/dev/null; do sleep 30; done
fi
echo "GPU free; starting K21i (epochs 20)"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_seetalk_k22.py \
  --d 128 --epochs 20 --n 60000 --device gpu --lam-concept 0.6 --seed 0 --out seetalk_k21i