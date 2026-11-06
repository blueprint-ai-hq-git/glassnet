#!/bin/bash
# The two cross-corpus measurements, both in the model's own vocabulary so they are comparable:
#  1. the RECORDED splitfix model (63.85% on its own corpus) measured on the MERGED corpus's held-out text
#  2. our current merged-trained small model measured on the SPLITFIX corpus's held-out text
# If (1) scores high on the merged text, the merged data is fine and our current models are the problem.
# If (1) collapses too, the merged corpus's held-out text is intrinsically harder/different.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
exec > /tmp/cross_eval.log 2>&1
echo "=== CROSS EVAL $(date '+%F %H:%M')"
echo "########## recorded splitfix model (63.85% own corpus) on MERGED held-out text"
bash lens/scripts/run_cross_eval.sh lens/runs/chat/t12_splitfix_d512_ep192.npz \
  "$PWD/data/chat/prepared_splitfix" "$PWD/data/chat/prepared_ext" SPLITFIXship_ON_mergedval
echo "########## the same model on its OWN corpus (control: must reproduce ~63.85%)"
bash lens/scripts/run_cross_eval.sh lens/runs/chat/t12_splitfix_d512_ep192.npz \
  "$PWD/data/chat/prepared_splitfix" "$PWD/data/chat/prepared_splitfix" SPLITFIXship_OWNcorpus
echo "########## our merged-trained small model on SPLITFIX held-out text"
bash lens/scripts/run_cross_eval.sh lens/runs/chat/sb_S0_d256_noattn_ep96.npz \
  "$PWD/data/chat/prepared_ext" "$PWD/data/chat/prepared_splitfix" MERGEDsmall_ON_splitfixval
echo "=== DONE $(date '+%F %H:%M')"