#!/bin/bash
# run_ours_sweep_part2.sh — 2026-10-24 self-wake: PART 2 of our method's settings sweep, the SIX arms that the
# seven-arm runner never reached because it was CUT AT THE ARM-A BOUNDARY (reason and numbers:
# lens/runs/SWEEP-CUT-PREDICTIONS-P90.txt; the cut itself, the verification: lens/runs/SWEEP-CUT-VERIFY.txt).
#
# WHY A NEW FILE AND NOT AN EDIT: two turns editing one runner is how experiments got mixed on this project
# before. run_ours_sweep.sh is left exactly as it was found and is NOT re-run - arm A is already complete
# (2 epochs; its record is lens/runs/chat/ours_arm_A_ours_tol0_history.jsonl).
#
# PREDICTIONS: unchanged and still owed. They are written at the top of lens/scripts/run_ours_arm.sh, which every
# arm below calls. Arm A's answer so far, from its own files: 12.29% by position at epoch 1 against the twin's
# 32.52% (lens/runs/chat/ours_arm_A_ours_tol0_history.jsonl vs ct_d256_full169_fix3_history.jsonl) with the gap
# gate refusing 77.9% of its checks, so 1,197 of 7,813 steps moved the weights. Prediction 2 of that file - does
# loosening the tolerance raise held-out and lower the refusals - is still the OPEN question, so the tolerance
# arms run first; the matched 2M-window control (G) runs first of all because it is the cheap reference that
# makes the others readable; the variant arms (D, E) run last.
#
# COST, MEASURED on arm A: 4,670 s per epoch at 2 cores -> ~2.6 h per ours-arm, ~15 h for the six below, and
# arm G (plain loop) is the fast one. This line sits BEHIND the critical path in QUEUE.txt on purpose.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
echo "---------------------------------------------------------------" >> lens/runs/OURS-SWEEP.txt
echo "SWEEP PART 2 (arms B-G, arm A already complete) $(date '+%F %H:%M')" >> lens/runs/OURS-SWEEP.txt
bash lens/scripts/run_ours_arm.sh G_standard --method standard >> lens/runs/OURS-SWEEP.txt 2>&1
bash lens/scripts/run_ours_arm.sh B_ours_tol001 --method ours --gate-tol 0.01 >> lens/runs/OURS-SWEEP.txt 2>&1
bash lens/scripts/run_ours_arm.sh C_ours_tol005 --method ours --gate-tol 0.05 >> lens/runs/OURS-SWEEP.txt 2>&1
bash lens/scripts/run_ours_arm.sh F_ours_lr002 --method ours --opt-lr 0.02 >> lens/runs/OURS-SWEEP.txt 2>&1
bash lens/scripts/run_ours_arm.sh D_ours_ratioratio --method ours --gate-variant rate_ratio >> lens/runs/OURS-SWEEP.txt 2>&1
bash lens/scripts/run_ours_arm.sh E_ours_bestband --method ours --gate-variant best_band >> lens/runs/OURS-SWEEP.txt 2>&1
echo "SWEEP PART 2 DONE $(date '+%F %H:%M')" >> lens/runs/OURS-SWEEP.txt
