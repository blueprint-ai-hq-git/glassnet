#!/bin/bash
# run_code_fairtest.sh — the clean-prompt test on the checkpoint we ALREADY have.
#
# Why first: it needs no training, so it separates two questions immediately.
#   Q1. is the 6/25 low because the model is bad, or because 20/25 prompts hand it `def <unk> (`?
#   Q2. does the loss-mask change (--drop-unk) add anything on top?
#
# Q1 is answered here, on covocab (32k vocab), which is already on disk.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CODE_DOC=code
export CD_VOCAB=32000

echo "############ FAIR TEST on the EXISTING 32k checkpoint (no retraining) ############"
CKPT=covocab_ep16 CD_VOCAB=32000 CODE_DOC=code nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
  lens/scripts/reexec_code.py 2>&1 | tail -45
