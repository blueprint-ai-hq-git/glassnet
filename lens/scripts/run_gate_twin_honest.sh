#!/bin/bash
# run_gate_twin_honest.sh — the gate arm's UNGATED TWIN, re-scored on the HONEST by-position column.
#
# WHY: the running gate arm (tag gate_sgd04_s2) is the stored recipe opt_sgd04_s2 with ONE change (--gates gap).
# The stored twin was trained on 2026-10-21, BEFORE the assistant-mask fix, so its history rows carry only the
# LEGACY by-token-id columns (val_masked_acc 0.2178 at ep1, 0.28464 final) and NO val_asst_acc. Its checkpoints
# are on disk, so the honest column for the twin can be MEASURED instead of guessed.
#
# WHAT IT IS: the P108 instrument (lens/scripts/dialogue_mixed_eval.py) unchanged, pointed at three checkpoints
# by CKPT_LIST; it uses the trainer's own val windows and the by-position assistant mask, and prints its own
# control (measured vs each checkpoint's STORED val_asst_acc) where a stored number exists.
# SAFE BESIDE THE LIVE TRAINER: one process, CPU, nice 19, 2 BLAS threads, read-only memmap of the corpus.
#
# OUTPUT: appended to lens/runs/GATE-TWIN-HONEST.txt (its own file, never overwritten).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2

CKPT_LIST="gate_arm_ep1=lens/runs/chat/gate_sgd04_s2_best.npz;twin_opt_sgd04_s2_ep1=lens/runs/chat/opt_sgd04_s2_ep1.npz;twin_opt_sgd04_s2_ep12=lens/runs/chat/opt_sgd04_s2_ep12.npz"

{
  echo ""
  echo "======================================================================================================"
  echo "GATE ARM vs ITS UNGATED TWIN ON THE HONEST COLUMN — $(date '+%F %H:%M:%S')  [bash lens/scripts/run_gate_twin_honest.sh]"
  echo "predictions: lens/runs/GATE-ARM-GRADE-PREDICTIONS-P109.txt (written before this runner existed)"
  echo "checkpoints: $CKPT_LIST"
  echo "NOTE the declared confound: the twin trained BEFORE the assistant-mask fix, so its weights come from the"
  echo "legacy objective. The honest column is computed identically for both; the twin's row is therefore a"
  echo "reference of unknown-but-declared bias, not a clean control arm."
  echo "======================================================================================================"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" \
  EVAL_DEVICE=cpu \
  DM_SCOPE="self-wake gate-arm-vs-twin P109" \
  CKPT_LIST="$CKPT_LIST" \
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/dialogue_mixed_eval.py
  echo "TWINHONEST EXIT=$?  $(date '+%F %H:%M:%S')"
} >> lens/runs/GATE-TWIN-HONEST.txt 2>&1
