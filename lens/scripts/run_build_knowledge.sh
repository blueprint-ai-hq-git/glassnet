#!/bin/bash
# Build the knowledge/maths corpus, then make its held-out split disjoint from train.
# Runs on CPU (pyarrow + numpy only) so it does not compete with a GPU training run for VRAM, and at
# nice 19 so the desktop stays responsive. Detached: it takes a while and must survive the agent's
# process-tree cleanup (see the glassnet skill, references/long-run-lifecycle.md).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG=/tmp/build_knowledge.log
{
  echo "=== BUILD KNOWLEDGE CORPUS $(date '+%F %H:%M')"
  nice -n 19 ionice -c3 .venv-data/bin/python3 -u lens/scripts/build_knowledge_corpus.py
  echo "build exit=$?"
  echo
  echo "=== DISJOINT SPLIT REPAIR (same rule as the other honest corpora)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/repair_split.py \
    data/chat/prepared_kn data/chat/prepared_kn_honest 32
  echo "repair exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
  du -sh data/chat/prepared_kn data/chat/prepared_kn_honest 2>/dev/null
} > "$LOG" 2>&1
tail -25 "$LOG"