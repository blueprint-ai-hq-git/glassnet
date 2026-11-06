#!/bin/bash
# run_record_library.sh — record the library suite results, then confirm nothing is running.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
".venv-gpu/bin/python3" -u scripts/record_library.py 2>&1 | tail -4
cp STATUS.md /home/yahya/GLASSNET-STATUS.md
echo
echo "=== nothing left running that needs the owner ==="
pgrep -af "t3_models.py|f_train.py|f_audit.py|b_grid.sh|test_library" | head -5 || echo "  no GlassNet jobs running"
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== STATUS.md numbered lines ==="
grep -cE "^[0-9]+\." STATUS.md