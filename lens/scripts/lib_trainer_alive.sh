#!/bin/bash
# Shared detector: is a real train_chat.py RUN actually executing?
#
# Why this file exists (2026-11-06 17:59, MEASURED): run_guard_e2e.sh refused to start with
# "BLOCKED: another train_chat.py is alive" while no trainer existed. The cause was
# `pgrep -af "train_chat.py"`, which matches any process whose COMMAND LINE contains that string -
# including a hermes agent whose -z prompt quotes the filename (pid 3301646 at the time). The reverse
# false negative is the same bug: a guard that can be fooled by prose is not a guard.
#
# The trainer is always started as:
#   .venv-gpu/bin/python3 -u lens/scripts/train_chat.py <args...>
# so the detector anchors the interpreter at the START of the command line and requires the exact
# script path after -u. An agent prompt can contain the words but cannot start that way.
#
# Usage:  source lens/scripts/lib_trainer_alive.sh ; if trainer_alive; then ... ; fi
#         source lens/scripts/lib_trainer_alive.sh ; trainer_alive_print   # human-readable list

TRAINER_RE='^(\./)?(\.venv-gpu/bin/python3|[^ ]*/python3|python3) -u lens/scripts/train_chat\.py( |$)'

trainer_alive() {
  pgrep -f "$TRAINER_RE" >/dev/null 2>&1
}

trainer_alive_print() {
  # -a prints pid + full command line, so the caller can log exactly what blocked it
  pgrep -af "$TRAINER_RE" 2>/dev/null
}
