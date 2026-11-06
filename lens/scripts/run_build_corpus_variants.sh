#!/bin/bash
# CORPUS VARIANTS for the comparison matrix (equal-budget runs, then pick the winner).
# Variant "chat": hh + oasst1 + ultrachat (no math) -> data/chat/prepared_ext_chat
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export EXT_OUT="$PWD/data/chat/prepared_ext_chat"
export EXT_SOURCES="hh,oasst1,ultrachat"
# shared vocab so all matrix runs are comparable on ONE held-out val
export EXT_VOCAB_FROM="$PWD/data/chat/prepared_ext"
.venv-data/bin/python3 -u lens/scripts/build_external_chat_corpus.py 2>&1 | tail -8
echo "=== variants on disk:"
du -sh data/chat/prepared_ext data/chat/prepared_ext_chat 2>/dev/null