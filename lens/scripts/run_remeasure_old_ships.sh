#!/bin/bash
# RE-MEASURE THE OLD SHIPS ON A VAL SET THAT IS LEAK-FREE FOR THEM (blast radius of the contamination
# finding). The recorded 63.85% came from prepared_clean's val, which is 63.21% verbatim-present in the
# splitfix model's TRAINING text. Each ship must be measured on the val of the corpus it trained on.
#   t12_splitfix (trained on prepared_splitfix) -> prepared_splitfix val   [already measured: 39.69%]
#   t11_talk41   (trained on the original prepared corpus, split unknown)  -> prepared val (NOTE: the
#                 original val split is known to be ~22.2% leaky, so its number is still mildly inflated
#                 and is reported with that caveat)
#   t12_d512_full (raw split, original corpus) -> prepared val
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
echo "=== t12_splitfix on prepared_splitfix val (the honest in-distribution number)"
EVAL_DATA_DIR="$PWD/data/chat/prepared_splitfix" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/t12_splitfix_d512_ep192.npz 2>&1 | grep -E "CLEAN VAL|assistant-masked"
echo
echo "=== t11_talk41 (recorded 54.89%) on the ORIGINAL prepared val"
echo "    caveat: the original val split is ~22.2% leaky, so this number is still mildly inflated"
EVAL_DATA_DIR="$PWD/data/chat/prepared" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/t11_talk41_ep192.npz 2>&1 | grep -E "CLEAN VAL|assistant-masked"
echo
echo "=== t12_d512_full (recorded 56.07%) on the ORIGINAL prepared val"
EVAL_DATA_DIR="$PWD/data/chat/prepared" bash lens/scripts/run_eval_clean.sh \
  lens/runs/chat/t12_d512_full_ep192.npz 2>&1 | grep -E "CLEAN VAL|assistant-masked"