#!/bin/bash
# finish_l3.sh — record L3, copy key files, report the session totals.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_l3.py 2>&1 | tail -3
cp STATUS.md REBUILD-PLAN.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo "copied to ~"
echo
echo "=== lens tree ==="
find lens -name "*.py" -o -name "*.c" -o -name "*.cu" | sort | sed 's/^/  /'
echo
echo "=== gates ==="
for g in test_l0 test_l1 test_l2_smoke test_l2_ops test_l3; do
  [ -f "lens/tests/$g.py" ] && echo "  lens/tests/$g.py"
done
echo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader