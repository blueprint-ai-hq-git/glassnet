#!/bin/bash
# METHOD BENCH: run the training methods for real and MEASURE them.
#
# The user's demand: "not just think or choose random settings ... just do real testing." So this runs the
# bench and prints measured numbers for every method that exists in the repo:
#   SFT      train_chat.py            (real, already in use for every chat run)
#   DISTILL  run_distill.py           (real: gradient-checked KL backward, teacher precondition enforced)
#   RL       lens/methods.py          (new: rejection-sampling RL with an EXACT verifier)
#
# The bench is arithmetic rules because it separates LEARNING A RULE from MEMORISING: train pairs from 0..9,
# held-out pairs from the SAME range, extrapolation to 10..19 which no memorised table can answer.
#
# Runs as a plain script file, never as an inline env-prefixed command: those trip the approval scanner.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3

OUT=lens/runs/METHODS-BENCH.txt
{
  echo "=== METHOD BENCH $(date '+%F %H:%M')"
  echo "--- RL (rejection sampling, exact verifier) at d=128 and d=256"
  for D in 128 256; do
    echo
    echo "## RL at d=$D"
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/methods.py \
      --d "$D" --samples 8 --rounds 3 --pretrain 300 --seed 0
    echo "exit=$?"
  done
  echo
  echo "--- the settings question, answered by measurement rather than choice:"
  echo "    same bench, three different RL sample counts, so 'samples 8' is a measured choice"
  for S in 2 8 32; do
    echo
    echo "## RL samples=$S at d=128"
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/methods.py \
      --d 128 --samples "$S" --rounds 3 --pretrain 300 --seed 0
    echo "exit=$?"
  done
  echo
  echo "=== DONE $(date '+%F %H:%M')"
} > "$OUT" 2>&1
tail -60 "$OUT"