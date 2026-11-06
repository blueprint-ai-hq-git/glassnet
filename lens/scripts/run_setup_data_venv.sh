#!/bin/bash
# A SEPARATE venv for DATA I/O only (parquet reading). Reason: .venv-gpu holds the validated
# numpy/CuPy stack that every training run depends on -- installing readers there risks changing
# the stack mid-experiment. This venv never trains anything; it only converts data.
set -e
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
if [ ! -d .venv-data ]; then
  python3 -m venv .venv-data
fi
.venv-data/bin/pip install --quiet --upgrade pip
.venv-data/bin/pip install --quiet pyarrow
.venv-data/bin/python3 -c "import pyarrow; print('pyarrow', pyarrow.__version__, 'ready in .venv-data')"