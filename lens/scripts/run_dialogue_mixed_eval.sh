#!/bin/bash
# run_dialogue_mixed_eval.sh — RUNNER for lens/scripts/dialogue_mixed_eval.py (predictions P108a-h).
# Predictions were written BEFORE the instrument file existed:
#   lens/runs/DIALOGUE-VS-MIXED-PREDICTIONS-P108.txt
#
# WHAT IT PRINTS: what a stored checkpoint scores on the DIALOGUE part of its held-out set versus the
# maths/textbook/code part, on the trainer's own assistant-by-position positions, plus the population count of
# each part. Measurement only: no training, no generation, no engine edits, every input read-only.
#
# OUTPUT: appended (never overwritten) to lens/runs/DIALOGUE-VS-MIXED.txt
# ONE PROCESS, CPU, nice 19, 2 BLAS threads: safe beside a GPU trainer (it holds a read-only memmap of the
# corpus, not a private copy), but it must not be run twice at once.
#
# CKPT_LIST="label=path;label2=path2" overrides the default set; relative paths are taken from the project root.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2

CKPT_LIST="${CKPT_LIST:-twin_d256_3ep=lens/runs/chat/ct_d256_full169_fix3_best.npz;h2_noconcept_3ep=lens/runs/chat/h2_noconcept_169_3_best.npz;h5l_6ep=lens/runs/chat/ct_d256_full169_6ep_best.npz;h2l_noconcept_6ep=lens/runs/chat/h2l_noconcept_169_6ep_best.npz}"

{
  echo ""
  echo "======================================================================================================"
  echo "DIALOGUE vs MIXED HELD-OUT EVAL — RUNNER $(date '+%F %H:%M:%S')  [bash lens/scripts/run_dialogue_mixed_eval.sh]"
  echo "predictions: lens/runs/DIALOGUE-VS-MIXED-PREDICTIONS-P108.txt (written before the instrument existed)"
  echo "checkpoints: $CKPT_LIST"
  echo "======================================================================================================"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" \
  EVAL_DEVICE="${EVAL_DEVICE:-cpu}" \
  DM_SCOPE="self-wake dialogue-vs-mixed P108" \
  CKPT_LIST="$CKPT_LIST" \
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/dialogue_mixed_eval.py
  echo "DM EXIT=$?  $(date '+%F %H:%M:%S')"
} >> lens/runs/DIALOGUE-VS-MIXED.txt 2>&1
