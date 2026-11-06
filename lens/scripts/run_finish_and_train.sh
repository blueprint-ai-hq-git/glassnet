#!/bin/bash
# Finish the combined corpus and then train on it. Replaces the 4G-capped run that was killed mid-step.
#
# ROOT CAUSE this fixes: the ChatML control-token step died with no exit line because the scope's
# MemoryMax=4G was below what a 294M-token rewrite needs, so the kernel killed it inside its own cgroup and
# every downstream step aborted for want of a corpus. Two changes: a realistic cap, and an explicit exit
# code printed after EVERY step so a silent kill is impossible to miss again.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=4
OUT=lens/runs/FINISH-AND-TRAIN.txt
PY=.venv-gpu/bin/python3
step () {                      # run one step, print its exit code, stop the chain if it failed
  echo "--- $1  $(date '+%H:%M')"
  shift
  nice -n 19 ionice -c3 "$@"
  rc=$?
  echo "exit=$rc"
  [ "$rc" -eq 0 ] || { echo "## CHAIN STOPPED at the step above (exit $rc) -- the real error is above"; return 1; }
  return 0
}
{
  echo "=== FINISH AND TRAIN $(date '+%F %H:%M')"
  echo "free memory at start: $(free -g | awk 'NR==2{print $3"G used of "$2"G"}')"
  step "1/5 control tokens (ChatML) on the merged corpus" \
    $PY -u lens/scripts/add_control_tokens.py data/chat/prepared_mix data/chat/prepared_mix_ctrl || exit 1
  step "2/5 base split for the control-token corpus" \
    $PY -u lens/scripts/make_split.py data/chat/prepared_mix_ctrl 0.02 0 || exit 1
  step "3/5 make held-out disjoint from train at the training window" \
    $PY -u lens/scripts/repair_split.py data/chat/prepared_mix_ctrl data/chat/prepared_mix_ctrl_honest 32 || exit 1
  echo "corpus built:"; du -sh data/chat/prepared_mix_ctrl_honest
  echo "--- 4/5 small smart model on the combined corpus"
  bash lens/scripts/run_mix_small.sh
  echo "exit=$?"
  echo "--- 5/5 big rungs d=1536/2048/3072"
  bash lens/scripts/run_ladder_big.sh
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
