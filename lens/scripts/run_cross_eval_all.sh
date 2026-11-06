#!/bin/bash
# HONEST CROSS-CORPUS EVALS (v2): one vocabulary per comparison, repaired splits, and every evaluation
# window that appears verbatim in the ship's OWN TRAINING text removed. Each run prints how many windows
# were removed, so the size of the correction is visible rather than assumed.
# (v1, which measured raw numbers without the filter, is kept as run_cross_eval_all_v1_raw.sh.)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
exec > /tmp/cross_eval.log 2>&1
B="$PWD/data/chat"
echo "=== HONEST CROSS EVAL $(date '+%F %H:%M')"

echo "########## 1. recorded splitfix ship on its OWN honest val (in-distribution reference)"
bash lens/scripts/run_cross_eval.sh lens/runs/chat/t12_splitfix_d512_ep192.npz \
  "$B/prepared_splitfix_honest" "$B/prepared_splitfix_honest" SPLITFIXship_ownHONEST \
  "$B/prepared_splitfix_honest"

echo "########## 2. recorded splitfix ship on MERGED honest val (cross-corpus, filtered)"
bash lens/scripts/run_cross_eval.sh lens/runs/chat/t12_splitfix_d512_ep192.npz \
  "$B/prepared_splitfix_honest" "$B/prepared_ext_honest" SPLITFIXship_onMERGED_HONEST \
  "$B/prepared_splitfix_honest"

echo "########## 3. our merged-trained small arm on SPLITFIX honest val (cross-corpus, filtered)"
bash lens/scripts/run_cross_eval.sh lens/runs/chat/sb_S0_d256_noattn_ep96.npz \
  "$B/prepared_ext_honest" "$B/prepared_splitfix_honest" MERGEDsmall_onSPLITFIX_HONEST \
  "$B/prepared_ext_honest"

echo "=== DONE $(date '+%F %H:%M')"