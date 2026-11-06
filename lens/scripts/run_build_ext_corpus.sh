#!/bin/bash
# Build the merged external talk corpus (HH-RLHF + oasst1 + ultrachat + orca-math).
# Runs in the DATA venv (pyarrow), never touches the training env.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-data/bin/python3 -u lens/scripts/build_external_chat_corpus.py > /tmp/build_ext_corpus.log 2>&1
echo "RC=$?"; tail -14 /tmp/build_ext_corpus.log