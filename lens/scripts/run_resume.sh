#!/bin/bash
# ONE-COMMAND RESUME after a stop. Runs the chains in order; every chain now skips finished arms and
# resumes partial ones from their newest checkpoint, so nothing starts over from epoch 1.
#   corpus arms (S2/S3 already done -> skipped)  ->  full-schedule arms (F1 done -> skipped, F2 resumes)
#   ->  H-block (waits for a quiet GPU, then H1..H5)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
echo "=== RESUME started $(date '+%F %H:%M')"
echo "=== nothing is deleted; finished arms are skipped, partial arms continue from their checkpoint"
bash lens/scripts/run_corpus_ab.sh
bash lens/scripts/run_full_schedule.sh
bash lens/scripts/run_h_block.sh
echo "=== RESUME chain finished $(date '+%F %H:%M')"