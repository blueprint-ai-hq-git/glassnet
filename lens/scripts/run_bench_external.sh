#!/bin/bash
# Run the external benchmark harness on CPU (the GPU is busy with the attention experiment).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
# CRITICAL: clear PYTHONPATH/PYTHONHOME. When this script is launched from a Hermes-managed
# process, Hermes's own venv leaks in through PYTHONPATH and its numpy (py3.14 build) shadows the
# project venv's numpy -> "Importing the numpy C-extensions failed". Unsetting keeps .venv-gpu's
# stack authoritative.
unset PYTHONPATH PYTHONHOME
export BENCH_DEVICE=cpu
export BENCH_SHIP="${BENCH_SHIP:-lens/runs/chat/t12_splitfix_d512_ep192.npz}"
export BENCH_VOCAB="${BENCH_VOCAB:-data/chat/prepared/vocab_v12000.json}"
export BENCH_N_MMLU="${BENCH_N_MMLU:-500}"
export BENCH_N_GSM8K="${BENCH_N_GSM8K:-200}"
export BENCH_N_CODE="${BENCH_N_CODE:-100}"
.venv-gpu/bin/python3 -u lens/scripts/bench_external.py > /tmp/bench_external.out 2>&1
echo "RC=$?"
grep -E "ship:|vocab:|=== |    [a-z_]+:|FAILED" /tmp/bench_external.out | head -30