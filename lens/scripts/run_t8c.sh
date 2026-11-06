#!/bin/bash
# T8c (code line, final): ONE change from the T8 run — concept set 35 -> 29 (DROP the 6
# strict-weak: reads_input, uses_range, mentions_a_list, mentions_each_element, has_slicing,
# has_a_call — all low-own-corr 0.08-0.31 with margins 0.04-0.09; token formulas cannot be
# sharpened further; never-forced rule). Epochs BACK to 32 (T8b's 64 hurt val 40.26->37.39
# while g1 honesty 27->29; the 6 weak add no accuracy).
# BAR (pre-registered, binary): STRICT audit on the kept set = 100% (n/n) AND val_acc >= 39
# AND syntax at n=150 >= 31 (no regression vs t8). Pre-gate <=0.2 recorded as informational
# for code (dense windows make >0.2 pairs unavoidable; the strict audit is the ship gate).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CD_VOCAB=32000 CODE_DOC=clean
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 32 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --out t8c_code29 \
  --max-windows 1200000