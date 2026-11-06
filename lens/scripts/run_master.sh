#!/bin/bash
# MASTER RUNNER - everything, in the right order, one GPU job at a time.
#
# WHY A MASTER: on 2026-10-18 17:22 a machine-wide OOM kill took down the agent's whole process group and
# killed every job I had started, because they all lived in the gateway's cgroup. This runs the same work
# as ONE job inside its own systemd scope with a memory cap, so (a) a gateway failure cannot kill it and
# (b) if it ever overreaches on memory, the kernel kills IT, not the desktop.
#
# Order, deliberately:
#   1. combined corpus build   (RAM-heavy, CPU only, runs ALONE)
#   2. baseline size sweep     (d=128 -> 256 -> 512, resuming interrupted sizes)
#   3. control-token ladder    (d=64 -> 1024, standard chat format, random draws at every rung)
#   4. small smart model       (d=256 on the combined corpus, smartness ruler + all gates)
#   5. big rungs               (d=1536 / 2048 / 3072, inside the 50M parameter budget)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
LOG=/tmp/master_run.log
{
  echo "=== MASTER RUN $(date '+%F %H:%M')"
  echo "--- 1/5 combined corpus build"
  bash lens/scripts/run_mix_build.sh
  echo "--- 2/5 baseline size sweep"
  bash lens/scripts/run_size_sweep.sh
  echo "--- 3/5 control-token ladder"
  bash lens/scripts/run_ladder_ctrl.sh
  echo "--- 4/5 small smart model on the combined corpus"
  bash lens/scripts/run_mix_small.sh
  echo "--- 5/5 big rungs"
  bash lens/scripts/run_ladder_big.sh
  echo "=== MASTER RUN DONE $(date '+%F %H:%M')"
  touch lens/runs/MASTER-DONE.flag      # persistent: /tmp is not trustworthy for this
} >> "$LOG" 2>&1
