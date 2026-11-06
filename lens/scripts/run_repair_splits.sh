#!/bin/bash
# Build the honest versions of the corpora: val disjoint from train at the 32-token training window size.
# Non-destructive: new directories, originals untouched.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
exec > /tmp/repair_splits.log 2>&1
echo "=== SPLIT REPAIR $(date '+%F %H:%M')"
for c in prepared_ext prepared_splitfix prepared_clean; do
  echo "########## $c"
  .venv-gpu/bin/python3 -u lens/scripts/repair_split.py \
    "$PWD/data/chat/$c" "$PWD/data/chat/${c}_honest" 32
done
echo "=== DONE $(date '+%F %H:%M')"
ls -d data/chat/*_honest 2>/dev/null