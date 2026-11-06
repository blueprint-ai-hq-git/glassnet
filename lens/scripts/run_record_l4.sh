#!/bin/bash
# run_record_l4.sh
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_l4.py 2>&1 | tail -3
cp STATUS.md LOG.md REBUILD-PLAN.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo copied
echo
echo "=== every gate, on record ==="
for g in test_l0 test_l1 test_l2_smoke test_l2_ops test_l2_train test_l2_scale test_l3 test_l4; do
  [ -f "lens/tests/$g.py" ] && echo "  lens/tests/$g.py"
done
echo
echo "=== jobs ==="
ps -eo pid,cmd | grep -E "test_l|g2_scaling" | grep -v grep || echo "  nothing running"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader