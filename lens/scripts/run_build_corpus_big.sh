#!/bin/bash
# BIG-MIX corpus variant: every licensed source, capped per source, shared vocab with the other
# variants so all matrix runs are comparable on ONE held-out val.
# Sources: hh + oasst1 + oasst2 + ultrachat + orca_math + OpenOrca + SlimOrca + MetaMathQA +
#          OpenMathInstruct-2   (all MIT / Apache-2.0 / CC-BY-4.0, recorded in DATA_LICENSES.md)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export EXT_OUT="$PWD/data/chat/prepared_ext_big"
export EXT_SOURCES="hh,oasst1,oasst2,ultrachat,orca_math,openorca,slimorca,metamath,openmath"
export EXT_VOCAB_FROM="$PWD/data/chat/prepared_ext"     # shared vocab -> comparable yardstick
# MEMORY SAFETY: this build previously OOM-killed the script (cgroup killed python at 4.1 GB
# anon-rss). The loaders now read parquet in batches / stream the big JSON array, and this cap makes
# a runaway build die on its OWN limit instead of inviting the system OOM killer -- which could
# otherwise take the GPU training run down with it.
ulimit -v 6000000
nice -n 19 ionice -c3 .venv-data/bin/python3 -u lens/scripts/build_external_chat_corpus.py 2>&1 \
  | tail -18