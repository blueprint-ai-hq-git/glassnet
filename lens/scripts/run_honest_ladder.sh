#!/bin/bash
# HONEST LADDER — every ship measured on val windows that are genuinely held out for it.
# Includes two CONTROLS: ships whose val is already leak-free should not move when the filter is on.
# If a "clean" corpus's number drops when the filter is enabled, that corpus was not clean.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export EVAL_DROP_TRAIN_OVERLAP=1

show () { grep -E "contamination filter|CLEAN VAL|assistant-masked" ; }

echo "=== t11_talk41 (recorded 54.89%) on ORIGINAL prepared val, overlap removed"
EVAL_DATA_DIR="$PWD/data/chat/prepared" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/t11_talk41_ep192.npz 2>&1 | show
echo
echo "=== t12_d512_full (recorded 56.07%) on ORIGINAL prepared val, overlap removed"
EVAL_DATA_DIR="$PWD/data/chat/prepared" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/t12_d512_full_ep192.npz 2>&1 | show
echo
echo "=== CONTROL: t12_splitfix on splitfix val (already leak-fixed at build; must stay ~39.69%)"
EVAL_DATA_DIR="$PWD/data/chat/prepared_splitfix" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/t12_splitfix_d512_ep192.npz 2>&1 | show
echo
echo "=== CONTROL: our current small arm on merged val (verified leak-free; must stay ~13.81%)"
EVAL_DATA_DIR="$PWD/data/chat/prepared_ext" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/sb_S0_d256_noattn_ep96.npz 2>&1 | show