#!/bin/bash
# run_ceiling.sh — the honest ceiling, plus a check on what chat data is actually reachable.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4

.venv-gpu/bin/python3 -u lens/scripts/ceiling.py

echo
echo "################################################################"
echo "# CHAT DATA — what is reachable, and under which license"
echo "################################################################"
echo "allowed licenses: MIT, Apache 2.0, CC0, CC BY 4.0  (owner's rule)"
echo
for u in \
  "https://huggingface.co/datasets/OpenAssistant/oasst1/raw/main/README.md" \
  "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/raw/main/README.md" \
  "https://huggingface.co/datasets/Anthropic/hh-rlhf/raw/main/README.md" \
  "https://raw.githubusercontent.com/OpenAssistant/oasst1/main/README.md" \
  "https://www.gutenberg.org/cache/epub/11/pg11.txt" ; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 20 -L "$u")
  echo "  HTTP $code  $u"
done
