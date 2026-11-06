#!/bin/bash
# run_test_library_alt.sh — the library suite with the interpreter it was written for.
# The .venv-gpu python has numpy+cupy but no matplotlib, so test_train_mini_explain_and_report
# failed on the picture-saving feature only. Check the project's normal python too.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
for PY in "/home/yahya/.hermes/hermes-agent/venv/bin/python3" "python3" ".venv-gpu/bin/python3"; do
  [ -x "$PY" ] || command -v "$PY" >/dev/null 2>&1 || continue
  echo "=== interpreter: $PY"
  "$PY" -c "import matplotlib, numpy; print('   matplotlib', matplotlib.__version__, '| numpy', numpy.__version__)" 2>&1 | head -2
done
echo
echo "=== full suite with the project python ==="
"/home/yahya/.hermes/hermes-agent/venv/bin/python3" -u tests/test_library.py 2>&1 | grep -E "PASS|FAIL|LIBRARY TESTS" | tail -12