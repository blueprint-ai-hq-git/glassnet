#!/bin/bash
# T8d (code line, final): ONE change from T8c — concept set 27 -> 25 (DROP the 2 strict-weak
# has_a_string, has_a_tuple: margins 0.048/0.077; token formulas cannot sharpen dense-code
# collisions further). The remaining 25 were all honest WITHIN the 27-set, so dropping can only
# remove rivals -> strict 25/25 expected (the same argument that held for talk's T11).
# BAR (pre-registered, binary): strict 25/25 (100%) AND val_acc >= 39 AND syntax >= 31.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CD_VOCAB=32000 CODE_DOC=clean
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 32 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --out t8d_code25 \
  --max-windows 1200000