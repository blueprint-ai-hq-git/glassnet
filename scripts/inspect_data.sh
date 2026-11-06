#!/bin/bash
# inspect_data.sh — what keys does tiny_data return, and how do we decode a window?
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "now: $(date '+%F %T %Z')"
for p in .venv-gpu/bin/python3 /home/yahya/.hermes/hermes-agent/venv/bin/python3 \
         /home/yahya/hermes-env/bin/python3; do
  [ -x "$p" ] && echo "found interpreter: $p"
done
PY=".venv-gpu/bin/python3"
[ -x "$PY" ] || PY=python3
"$PY" -c "
import sys
sys.path.insert(0, '/run/media/yahya/HARD DISK 4/GlassNet')
from glassnet import tiny_data
d = tiny_data.build(win=8, vocab_cap=1500)
print('tiny_data keys:', sorted(d.keys()))
for k in sorted(d.keys()):
    v = d[k]
    print(f'  {k}: {type(v).__name__} shape={getattr(v, \"shape\", None)}')
"