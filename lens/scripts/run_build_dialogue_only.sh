#!/bin/bash
# run_build_dialogue_only.sh — RUNNER for lens/scripts/build_dialogue_only_corpus.py (predictions P110a-h).
# Predictions were written BEFORE the build script existed: lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt
#
# WHAT IT BUILDS: data/chat/prepared_dialogue_only_honest = conversations 0..156,519 of
# data/chat/prepared_mix_ctrl_honest (the ext_honest group: hh-rlhf, oasst1, ultrachat, orca-math = HUMAN
# DIALOGUE, ~39.0M tokens), sliced, NOT re-tokenised: same ids, same boundaries, same split, same vocab file.
# It is the DATA half of the arm whose predictions are P110a-h; it trains nothing and loads no model.
#
# SAFE BESIDE A LIVE TRAINER: read-only memmap of the mix corpus, one 156 MB slice, NO window array is ever
# materialised, ONE process, cores 2-3 (the trainer holds 0-1), nice 19. It writes only into the NEW directory
# and refuses (exit 3) if that directory already exists.
#
# OUTPUT: appended (never overwritten) to lens/runs/DIALOGUE-ONLY-BUILD.txt
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2

{
  echo ""
  echo "======================================================================================================"
  echo "BUILD DIALOGUE-ONLY CORPUS — RUNNER $(date '+%F %H:%M:%S')  [bash lens/scripts/run_build_dialogue_only.sh]"
  echo "predictions: lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt (written before the build script existed)"
  echo "======================================================================================================"
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/build_dialogue_only_corpus.py
  echo "BUILD EXIT=$?  $(date '+%F %H:%M:%S')"
} >> lens/runs/DIALOGUE-ONLY-BUILD.txt 2>&1
tail -n 4 lens/runs/DIALOGUE-ONLY-BUILD.txt
