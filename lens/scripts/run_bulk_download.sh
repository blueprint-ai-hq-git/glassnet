#!/bin/bash
# Bulk license gate + download of every ALLOWED candidate. Runs in background (network bound),
# alongside the GPU experiment. Disk-guarded to 12 GB of new data.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-data/bin/python3 -u lens/scripts/bulk_license_and_download.py > /tmp/bulk_download.log 2>&1
echo "RC=$?"
tail -5 /tmp/bulk_download.log