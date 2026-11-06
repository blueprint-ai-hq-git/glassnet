#!/bin/bash
# BUILD THE COMBINED CORPUS: talk + maths + general knowledge + code, in the standard chat format.
#
# This is the fuel for the "small model that talks, codes and knows things" run. Every source passed the
# licence gate (MIT / Apache-2.0 / CC0 / CC BY 4.0) and is listed in DATA_LICENSES.md.
#
# Steps, in order, each verified before the next:
#   1. build the knowledge+maths+code corpus            -> data/chat/prepared_kn2
#   2. merge it with the chat corpus (same 12,000-word vocabulary, so ids are compatible; the merge
#      refuses to run if the vocabularies differ)       -> data/chat/prepared_mix
#   3. add the standard control tokens (ChatML)         -> data/chat/prepared_mix_ctrl
#   4. base split, then make val disjoint from train at the 32-token training window
#                                                       -> data/chat/prepared_mix_ctrl_honest
# CPU only (pyarrow/numpy) at nice 19, so it runs alongside a GPU job without fighting for VRAM.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG=/tmp/mix_build.log
exec > "$LOG" 2>&1
echo "=== COMBINED CORPUS BUILD $(date '+%F %H:%M')"

# wait for the code download to land (it is a detached job; the builder skips absent files, but a
# half-downloaded corpus would be worse than waiting)
for i in $(seq 1 120); do
  pgrep -f "[r]un_get_code_slice" > /dev/null || break
  sleep 30
done
echo "code download finished or not running; proceeding"

echo "--- step 1: knowledge + maths + code"
KN_OUT=data/chat/prepared_kn2 nice -n 19 ionice -c3 .venv-data/bin/python3 -u \
  lens/scripts/build_knowledge_corpus.py
echo "step 1 exit=$?"

echo "--- step 1b: base split for the new corpus (the merge needs one; the builder does not create it)"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/make_split.py data/chat/prepared_kn2 0.02 0
echo "step 1b exit=$?"

echo "--- step 2: merge with the chat corpus"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/merge_corpora.py \
  data/chat/prepared_mix data/chat/prepared_ext_honest data/chat/prepared_kn2
echo "step 2 exit=$?"

echo "--- step 3: standard control tokens (ChatML)"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/add_control_tokens.py \
  data/chat/prepared_mix data/chat/prepared_mix_ctrl
echo "step 3 exit=$?"

echo "--- step 4: split, then make val disjoint from train"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/make_split.py data/chat/prepared_mix_ctrl 0.02 0
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/repair_split.py \
  data/chat/prepared_mix_ctrl data/chat/prepared_mix_ctrl_honest 32
echo "step 4 exit=$?"

echo "--- sizes:"
du -sh data/chat/prepared_kn2 data/chat/prepared_mix data/chat/prepared_mix_ctrl \
       data/chat/prepared_mix_ctrl_honest 2>/dev/null
echo "=== DONE $(date '+%F %H:%M')"