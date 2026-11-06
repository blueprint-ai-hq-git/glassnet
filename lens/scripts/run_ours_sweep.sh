#!/bin/bash
# ONE THING CHANGED PER ARM. Every arm appends its own args and epoch lines to OURS-SWEEP.txt,
# so no number can be read without the setting that produced it.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
bash lens/scripts/run_ours_arm.sh A_ours_tol0 --method ours --gate-tol 0.0 >> lens/runs/OURS-SWEEP.txt 2>&1
bash lens/scripts/run_ours_arm.sh B_ours_tol001 --method ours --gate-tol 0.01 >> lens/runs/OURS-SWEEP.txt 2>&1
bash lens/scripts/run_ours_arm.sh C_ours_tol005 --method ours --gate-tol 0.05 >> lens/runs/OURS-SWEEP.txt 2>&1
bash lens/scripts/run_ours_arm.sh D_ours_ratioratio --method ours --gate-variant rate_ratio >> lens/runs/OURS-SWEEP.txt 2>&1
bash lens/scripts/run_ours_arm.sh E_ours_bestband --method ours --gate-variant best_band >> lens/runs/OURS-SWEEP.txt 2>&1
bash lens/scripts/run_ours_arm.sh F_ours_lr002 --method ours --opt-lr 0.02 >> lens/runs/OURS-SWEEP.txt 2>&1
bash lens/scripts/run_ours_arm.sh G_standard --method standard >> lens/runs/OURS-SWEEP.txt 2>&1
