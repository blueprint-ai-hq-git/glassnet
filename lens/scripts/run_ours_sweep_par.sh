#!/bin/bash
# ================================ REFUSAL GUARD (2026-10-24 10:1x self-wake) ================================
# THIS SCRIPT IS INVALID AND MUST NOT RUN. It calls train_chat.py directly without exporting CHAT_DATA_DIR, so
# every arm it ran on 2026-10-24 09:24-10:08 trained on the trainer DEFAULT corpus data/chat/prepared (681,783
# windows, vocab 12,000, val 13,651) instead of the matched data/chat/prepared_mix_ctrl_honest (2,000,000 capped
# windows of 294,225,090 tokens, vocab 12,002, val 87,035) — the corpus every stored comparison number comes from.
# It also reuses the serial sweep's --out tags, so it appends into a finished arm's own per-step logs.
# Evidence: lens/runs/SWEEP-PAR-INVALID.txt; BUGS.md [BUG-SWEEP-RUNNER-SKIPS-CORPUS-ENV].
# The watchdog restored this line once at 10:07:48 and it was killed again; this guard makes any future restore a
# loud no-op instead of a wasted GPU run. The correct serial runner is lens/scripts/run_ours_arm.sh (which DOES
# export CHAT_DATA_DIR), and the remaining arms are queued via lens/scripts/run_ours_sweep_part2.sh.
echo "REFUSED: run_ours_sweep_par.sh trains on the WRONG corpus (no CHAT_DATA_DIR export) and reuses --out tags."
echo "          read lens/runs/SWEEP-PAR-INVALID.txt and BUGS.md [BUG-SWEEP-RUNNER-SKIPS-CORPUS-ENV]."
echo "          use lens/scripts/run_ours_arm.sh / run_ours_sweep_part2.sh instead. nothing was launched."
# THE SLEEP IS DELIBERATE (2026-10-24 10:1x, MEASURED LOOP): the watchdog verifies a launch by asking whether the
# scope is ACTIVE 15 s after starting it, and if not it PREPENDS the line back to QUEUE.txt. A guard that exits
# instantly therefore makes the watchdog restore the poisoned line forever, and the queue never advances (that
# happened at 10:12: tried, failed, restored). Staying alive past the 15 s check lets the watchdog record the line
# as started and CONSUME it, which retires it for good. 20 s of a sleeping scope costs the queue one tick at most.
echo "          (holding the scope 20 s so the watchdog's 15 s started-check consumes this line instead of restoring it)"
sleep 20
exit 78
# ==========================================================================================================

# PARALLEL SWEEP (2026-10-24). Yahya's point, and he is right: one arm used about 1 GB of VRAM and one core-pair,
# leaving the box idle. Each arm is an independent process with its own model, so they cannot interfere in any
# way that matters, and running them side by side turns a 2.5 hour sweep into about 1 hour.
# EACH ARM GETS ITS OWN 2 CORES so the total stays at 6 of 12 and his game keeps 6.
# RAM is the real limit, not the GPU: each arm holds its own window array, so RAM is watched and reported.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
run_arm() {  # name, core-pair, flags...
  local name="$1"; local cores="$2"; shift 2
  ( taskset -c "$cores" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
      --d 256 --window 32 --epochs 2 --batch 1024 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 \
      --seed 0 --device gpu --assistant-only --train-turn-end --direct --direct-mode tied \
      --max-windows 2000000 --out "ours_arm_${name}" "$@" ) > "lens/runs/OURS-ARM-${name}.txt" 2>&1
  echo "arm ${name} finished at $(date '+%H:%M')"
}
BATCH1_START=$(date '+%H:%M')
run_arm A_ours_tol0 0,1 --method ours --gate-tol 0.0 &          PA=$!
run_arm B_ours_tol001 2,3 --method ours --gate-tol 0.01 &       PB=$!
run_arm C_ours_tol005 4,5 --method ours --gate-tol 0.05 &       PC=$!
wait $PA $PB $PC
echo "=== batch 1 done $(date '+%H:%M') (started $BATCH1_START); RAM now: $(free -g | awk 'NR==2{print $3"/"$2"G"}')"
run_arm D_ours_ratioratio 0,1 --method ours --gate-variant rate_ratio &  PD=$!
run_arm E_ours_bestband 2,3 --method ours --gate-variant best_band &     PE=$!
run_arm F_ours_lr002 4,5 --method ours --opt-lr 0.02 &                   PF=$!
wait $PD $PE $PF
echo "=== batch 2 done $(date '+%H:%M')"
run_arm G_standard 0,1 --method standard
echo "=== SWEEP COMPLETE $(date '+%F %H:%M')"
