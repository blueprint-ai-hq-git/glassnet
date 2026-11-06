#!/bin/bash
# CPU / TINY-CORPUS WIRING PRE-FLIGHT for the drift discriminator (lens/scripts/run_drift_discriminator.sh).
# P107a-c are NOT answered here and the grader says so in its own text: the pre-flight's point is that the runner
# starts two runs, both write a row, the grader reads both rows and prints a table, and the whole thing exits 0.
# It is run on the TINY corpus and on the CPU on purpose: a train_chat process loads its WHOLE corpus cache
# ([BUG-TRAINCHAT-MEMORY-IS-THE-CORPUS]), and the real drift run must have the machine to itself.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "DRIFT PRE-FLIGHT — tiny corpus, CPU, d=64, 20,000 windows. NOT the discriminator. $(date '+%F %H:%M:%S')"
DRIFT_DATA_DIR="$PWD/data/chat/prepared_gate_small" \
DRIFT_PREFIX="drift_pf" \
DRIFT_DEVICE="cpu" \
DRIFT_D=64 \
DRIFT_WINDOWS=20000 \
DRIFT_EPOCHS=1 \
DRIFT_CKPT_SECONDS=0 \
DRIFT_LOG="$PWD/lens/runs/DRIFT-DISCRIMINATOR-PREFLIGHT.txt" \
DRIFT_REF="TWIN ct_d256_full169_fix3=$PWD/lens/runs/chat/ct_d256_full169_fix3_history.jsonl" \
taskset -c 4,5 nice -n 19 ionice -c3 bash lens/scripts/run_drift_discriminator.sh
echo "PRE-FLIGHT EXIT=$?  $(date '+%F %H:%M:%S')"
