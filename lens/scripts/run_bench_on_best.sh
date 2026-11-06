#!/bin/bash
# TRUE BASELINE RULER on our CURRENT BEST model (not the old ship).
# The numbers in BENCH_EXTERNAL.json were measured on t12_splitfix (an old, contaminated-score ship).
# Any "is it smart" claim must rest on the model we would actually ship, so this re-runs MMLU, GSM8K and
# the code test on sb_S0_d256_noattn_ep96 (the best honest talk model: 9.84% next-word).
# CPU only, nice 19. Writes lens/runs/BENCH_EXTERNAL_S0.json (the old file is left untouched).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export BENCH_DEVICE=cpu
export BENCH_SHIP="$PWD/lens/runs/chat/sb_S0_d256_noattn_ep96.npz"
export BENCH_VOCAB="$PWD/data/chat/prepared_ext_honest/vocab_v12000.json"
export BENCH_OUT="$PWD/lens/runs/BENCH_EXTERNAL_S0.json"
export BENCH_N_MMLU=500
export BENCH_N_GSM8K=200
export BENCH_N_CODE=100
echo "=== ruler on $(basename "$BENCH_SHIP") $(date '+%F %H:%M')"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/bench_external.py > /tmp/bench_S0.out 2>&1
echo "RC=$?"
grep -E "ship:|=== |acc|syntax|random_baseline|mean_prob|datum" /tmp/bench_S0.out | head -24