#!/bin/bash
# DECODE-ATTRACTOR runner (2026-10-22 20:2x self-wake). CPU ONLY, device cpu inside the instrument.
# Predictions P73a-i were written BEFORE this file ran: lens/runs/DECODE-ATTRACTOR-PREDICTIONS-P73.txt
# It writes ONLY lens/runs/DECODE-ATTRACTOR.txt (the instrument) and lens/runs/DECODE-ATTRACTOR-RUN.txt (this log).
# It does NOT touch any checkpoint, any stored run, lens/runs/SHIP-TALK-v3.txt, or the code it copied from
# (lens/scripts/talk_fair_v3_ship.py): the prompt selection and the decoder were COPIED into the new instrument.
# WHY CPU AND NOT THE GPU: the d=512 CRITICAL PATH 2 arm (pid 1494583, glassnet-q-3182532730-192758.scope) owns
# the GPU and the watchdog must not see a second trainer; this measurement needs no GPU at all.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export ATTRACTOR_OUT="$PWD/lens/runs/DECODE-ATTRACTOR.txt"
export CKPT_LIST="SHIP CANDIDATE d256 maskfix full-corpus FINAL best (ep3, held-out 35.26%)=lens/runs/chat/ct_d256_full169_fix3_best.npz;d512 full-corpus ep1 best (held-out 33.17% by position)=lens/runs/chat/ct_d512_full169_fix3_best.npz"
{
  echo "=== DECODE-ATTRACTOR runner $(date '+%F %H:%M:%S') ==="
  echo "NOTE: the FIRST launch of this runner (20:26:45) ran the whole d=256 half, then died on the SECOND"
  echo "      checkpoint with AttributeError: 'dict' object has no attribute 'to_dev'. Cause, found by reading this"
  echo "      machine's own code and NOT the model: the instrument bound a module-level name B (=lens.backend) at the"
  echo "      top and then re-used B as a local for the variant-B panel row, so the second load called to_dev on a"
  echo "      dict. Fixed by renaming those two variables (VA/VB). The d=256 numbers printed by that first launch are"
  echo "      kept in this log and are the numbers this file reports; the re-run re-measures them and prints the"
  echo "      verdict panel that the crash prevented. Nothing was overwritten: this log only ever appends."
  echo "checkpoints: $CKPT_LIST"
  for f in lens/runs/chat/ct_d256_full169_fix3_best.npz lens/runs/chat/ct_d512_full169_fix3_best.npz; do
    echo "  $(stat -c '%y %s bytes %n' "$f")"
  done
  echo "  predictions: lens/runs/DECODE-ATTRACTOR-PREDICTIONS-P73.txt"
  start=$(date +%s)
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/decode_attractor_test.py
  rc=$?
  echo "python exit=$rc   wall=$(( $(date +%s) - start ))s"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> lens/runs/DECODE-ATTRACTOR-RUN.txt 2>&1
