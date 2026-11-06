#!/bin/bash
# One-command customer pipeline test. Data = codeparrot apps (107 MB, MIT, already on disk) —
# deliberately a DIFFERENT distribution from the dialogue chat model was trained on: this is the
# "customer brings their own data" scenario.
# PRE-REGISTERED (fixed before running):
#   f1  holdout ppl < 2 x 3500 = 7000            (learns the customer's language, 2x better than guessing)
#   f2  clean ON and OFF both run; if ON's holdout ppl is WORSE than OFF by >10%, the cleaner is
#       damaging this data and that is reported as a cleaner finding (thresholds may not fit code text)
#   f3  counts reconcile stage by stage (the script hard-stops otherwise)
#   p1  prediction: apps text is long-ish prose+markdown; expect vocab 3500 mostly English words,
#       <unk> under 15%
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
for MODE in on off; do
  echo "############ --clean $MODE ############"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/feed_customer.py \
    --data data/code/apps_train.jsonl --clean $MODE --vocab 3500 --window 16 --d 128 \
    --epochs 2 --device gpu --max-train 200000 --out feed_apps 2>&1 | tail -16
done
