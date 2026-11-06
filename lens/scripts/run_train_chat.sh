#!/bin/bash
# run_train_chat.sh — train the talking model, then make it talk.
#
# Size chosen from the DATA, not from the GPU: 13.3M tokens supports a 1-3M parameter model
# adequately. A bigger model would be under-trained and would look worse for a reason that has
# nothing to do with the architecture.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 192 --hidden 512 --epochs "${TC_EPOCHS:-4}" --batch 256 --lr 0.4 \
  --device gpu --out "${TC_OUT:-run1}" 2>&1 | tail -70
