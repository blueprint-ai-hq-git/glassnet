#!/bin/bash
# REBUILD the control-token corpus, this time with COUNT VALIDATION at every step.
#
# THE BUG (2026-10-19): the control-token step was killed by a 4 GB memory cap last night and left a PARTIAL
# file: 13,318,447 tokens of the intended 294,057,312 (4.5%), silently switching int32 to int64. The morning
# chain skipped the step because the file EXISTED, so the small model and all three big rungs trained on a
# truncated slice. Existence is not validity. Every step below now checks the NUMBER, and a mismatch stops
# the chain instead of quietly training on bad data.
#
# Old results are never deleted: the partial corpus is MOVED aside, not removed.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=4
PY=.venv-gpu/bin/python3
OUT=lens/runs/REBUILD-CTRL.txt
{
  echo "=== REBUILD CONTROL CORPUS $(date '+%F %H:%M')"
  echo "memory at start: $(free -g | awk 'NR==2{print $3"G used of "$2"G"}')"

  SRC=data/chat/prepared_mix
  DST=data/chat/prepared_mix_ctrl
  SRC_TOK=$($PY -c "import numpy as np; print(len(np.load('$SRC/ids_v12000.npy', mmap_mode='r')))")
  echo "source $SRC has $SRC_TOK tokens"
  [ "$SRC_TOK" -gt 100000000 ] || { echo "## STOP: the source corpus itself looks truncated"; exit 1; }

  if [ -f "$DST/ids_v12000.npy" ]; then
    DST_TOK=$($PY -c "import numpy as np; print(len(np.load('$DST/ids_v12000.npy', mmap_mode='r')))")
    echo "existing $DST has $DST_TOK tokens"
    if [ "$DST_TOK" -lt "$SRC_TOK" ]; then
      echo "existing corpus is SHORTER than its source -> it is the partial file from the killed run."
      echo "moving it aside as ${DST}_partial_20261019 (nothing is deleted)"
      mv "$DST" "${DST}_partial_20261019"
    fi
  fi

  if [ ! -f "$DST/ids_v12000.npy" ]; then
    echo "--- building control tokens (ChatML) on the FULL merged corpus"
    nice -n 19 ionice -c3 $PY -u lens/scripts/add_control_tokens.py "$SRC" "$DST"
    echo "exit=$?"
  fi

  NEW_TOK=$($PY -c "import numpy as np; print(len(np.load('$DST/ids_v12000.npy', mmap_mode='r')))")
  NEW_CONV=$($PY -c "import numpy as np; print(len(np.load('$DST/bounds_v12000.npy', mmap_mode='r'))-1)")
  echo "rebuilt corpus: $NEW_TOK tokens, $NEW_CONV conversations"
  # the rewrite only INSERTS tokens, so the new corpus must be at least as long as the source, and the
  # conversation count must match. Anything else means a partial write and the chain must not proceed.
  if [ "$NEW_TOK" -lt "$SRC_TOK" ] || [ "$NEW_CONV" -ne 740665 ]; then
    echo "## STOP: rebuilt corpus fails validation (source $SRC_TOK tokens / 740665 conversations,"
    echo "## got $NEW_TOK / $NEW_CONV). Refusing to train on it."
    exit 1
  fi
  echo "VALIDATION PASSED: $NEW_TOK tokens (source $SRC_TOK + inserted control tokens), $NEW_CONV conversations"

  echo "--- split for the rebuilt corpus"
  nice -n 19 ionice -c3 $PY -u lens/scripts/make_split.py "$DST" 0.02 0 || exit 1
  echo "--- make val disjoint from train at the training window (streaming, memory-lean)"
  nice -n 19 ionice -c3 $PY -u lens/scripts/repair_split.py "$DST" data/chat/prepared_mix_ctrl_honest 32 || exit 1

  FIN=data/chat/prepared_mix_ctrl_honest
  FIN_TOK=$($PY -c "import numpy as np; print(len(np.load('$FIN/ids_v12000.npy', mmap_mode='r')))")
  FIN_CONV=$($PY -c "import numpy as np; print(len(np.load('$FIN/bounds_v12000.npy', mmap_mode='r'))-1)")
  echo "final corpus: $FIN_TOK tokens, $FIN_CONV conversations"
  [ "$FIN_TOK" -ge 200000000 ] || { echo "## STOP: final corpus is too small ($FIN_TOK tokens)"; exit 1; }
  echo "READY: training can now run on the real combined corpus"
  echo "=== DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
