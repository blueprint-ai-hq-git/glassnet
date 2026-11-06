#!/bin/bash
# What exists: the HH-RLHF source files, the external downloads' file layout, and whether parquet
# reading is available (needed to consume oasst1/ultrachat/orca-math).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "=== HH-RLHF source files:"; ls -la data/chat/*.gz data/chat/*.jsonl* 2>/dev/null | awk '{print $5, $9}'
echo "=== external downloads so far:"; du -sh data/external/* 2>/dev/null
echo "=== file lists:"; for d in oasst1 ultrachat orca_math gsm8k mmlu; do
  [ -d "data/external/$d" ] && echo "--- $d:" && find "data/external/$d" -type f | head -4
done
echo "=== parquet support:"; .venv-gpu/bin/python3 -c "import pyarrow; print('pyarrow', pyarrow.__version__)" 2>&1 | tail -1
echo "=== pandas:"; .venv-gpu/bin/python3 -c "import pandas; print('pandas', pandas.__version__)" 2>&1 | tail -1