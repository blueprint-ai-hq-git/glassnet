#!/bin/bash
# run_dialogue_only_preflight.sh — PRE-FLIGHT for the dialogue-only arm (predictions P110a-h).
#
# TWO THINGS, in this order:
#  1. the wiring pre-flight (lens/scripts/dialogue_only_preflight.py): the slice through the trainer's OWN
#     loader, the window/val arithmetic, every flag in the recipe against train_chat.py's argparse, the
#     best-epoch read block executed on a REAL history file, the chained deciders present, no ';' in a
#     CKPT_LIST label, no output file already existing.
#  2. A NUMERIC CONTROL ON THE SLICE ITSELF: the group-split instrument (lens/scripts/dialogue_mixed_eval.py,
#     the same one that produced lens/runs/DIALOGUE-VS-MIXED.txt) pointed at the SLICE instead of the mix, on
#     two STORED checkpoints. The slice holds ONE group, so its "ASST by position, ALL groups" column IS the
#     dialogue column, and the control is a reproduction: h2l must score 24.8029% with n = 15,728 and the twin
#     23.5313% with n = 15,728, to within the 0.0739-point drift floor [lens/runs/DRIFT-NUMBERS.txt].
#     IF THEY DO NOT REPRODUCE, THE SLICE IS NOT THE DIALOGUE SUBSET THE ARM ASSUMES AND THE ARM MUST NOT RUN.
#     NOTE, so nobody misreads it: the instrument's own P108a(ii) control compares each checkpoint against its
#     stored MIXED row (val_asst_acc 37.2480 / 35.2632), so it prints FAIL when pointed at the slice. That FAIL
#     is expected here and is not a finding; the reproduction of the dialogue column is the control.
#
# CPU ONLY, one process at a time, cores 2-3, nice 19, read-only inputs, no model is trained, nothing is
# written outside lens/runs/DIALOGUE-ONLY-PREFLIGHT.txt.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2

{
  echo ""
  echo "======================================================================================================"
  echo "DIALOGUE-ONLY ARM — PRE-FLIGHT $(date '+%F %H:%M:%S')  [bash lens/scripts/run_dialogue_only_preflight.sh]"
  echo "predictions: lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt (written before the corpus and the runner)"
  echo "======================================================================================================"

  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/dialogue_only_preflight.py
  echo "PREFLIGHT EXIT=$?  $(date '+%F %H:%M:%S')"

  echo ""
  echo "------------------------------------------------------------------------------------------------------"
  echo "NUMERIC CONTROL: the group-split instrument on the SLICE, on two stored checkpoints"
  echo "  expected: h2l_noconcept_6ep DIALOGUE column 24.8029% and twin 23.5313%, n = 15,728 each"
  echo "  (both from lens/runs/DIALOGUE-VS-MIXED.txt; floor 0.0739 pts, lens/runs/DRIFT-NUMBERS.txt)"
  echo "------------------------------------------------------------------------------------------------------"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_dialogue_only_honest" \
  EVAL_DEVICE=cpu \
  DM_SCOPE="DLONLY pre-flight numeric control: the slice as the corpus" \
  CKPT_LIST="h2l_noconcept_6ep=lens/runs/chat/h2l_noconcept_169_6ep_best.npz;twin_d256_3ep=lens/runs/chat/ct_d256_full169_fix3_best.npz" \
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/dialogue_mixed_eval.py
  echo "SLICE-CONTROL EXIT=$?  $(date '+%F %H:%M:%S')"

  echo ""
  echo "--- the arm's corpus, as the trainer will see it ---"
  ls -la data/chat/prepared_dialogue_only_honest/
  echo "=== PRE-FLIGHT DONE $(date '+%F %H:%M:%S') ==="
} >> lens/runs/DIALOGUE-ONLY-PREFLIGHT.txt 2>&1
tail -n 6 lens/runs/DIALOGUE-ONLY-PREFLIGHT.txt
