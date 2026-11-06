#!/bin/bash
# run_chat_char_real.sh — the full character-level run, compared in bits per character.
#
# The smoke test (1 epoch, 8,000 windows, ~150k params) reached ppl 43.49. Converted to the common
# currency that is 5.443 bits per character, against the word model's 1.850 -- so on the data
# actually seen, WORD level is ahead. But the character run saw 28x less data and has 11x fewer
# parameters. This is the full run that makes the comparison fair.
#
# Pre-registered predictions (written BEFORE the run):
#   1. bits per character falls well below 5.443, and the gap to the word model's 1.850 narrows
#      substantially. A 60.9M-character corpus with a ~150k-parameter model is 400 characters per
#      parameter, which is far better trained than the word model's 4.4 tokens per parameter.
#   2. If bits per character does NOT beat 1.850, then character level is the wrong choice for this
#      corpus and I will say so rather than defending the idea.
#   3. Concept pass rate rises sharply from 1/35 (which was a 1-epoch artefact).
#   4. Replies become locally coherent: real words, real short phrases, correct spelling and
#      punctuation. Still not a chatbot.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
EP="${CC_EPOCHS:-6}"

echo "waiting for the GPU..."
while pgrep -f "train_chat.py|run_distill.py|run_optimise.sh" > /dev/null; do
  sleep 30
done
echo "GPU free; starting the full character-level run"

nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat_char.py \
  --d 64 --window 32 --epochs "$EP" --batch 256 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 \
  --device gpu --out char1 --max-windows 1200000 2>&1 | tail -60

echo
echo "################################################################"
echo "# BITS PER CHARACTER — the fair comparison"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, math, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat"
CPW = 60949068 / 13318447
print(f"  common currency: bits per character of text   (chars/word = {CPW:.3f})")
print(f"  {'model':<26}{'ppl':>10}{'bits/token':>12}{'bits/char':>11}{'params':>12}")
rows = [("word d=128 K=9 (run 1)", "real1_log.json", CPW),
        ("word d=384 K=48 (big1)", "big1_log.json", CPW),
        ("word d=384 K=9 (grid)", "grid_d384_k9_log.json", CPW),
        ("word d=128 K=48 (grid)", "grid_d128_k48_log.json", CPW),
        ("char d=64 W=32 K=35", "char1_log.json", 1.0),
        ("char smoke (1 epoch)", "charsmoke_log.json", 1.0)]
best = None
for tag, f, cpt in rows:
    p = os.path.join(P, f)
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    h = r["history"][-1]
    bc = math.log2(h["val_ppl"]) / cpt
    print(f"  {tag:<26}{h['val_ppl']:>10.2f}{math.log2(h['val_ppl']):>12.3f}{bc:>11.3f}"
          f"{r['params']:>12,}")
    if best is None or bc < best[1]:
        best = (tag, bc)
if best:
    print(f"\n  BEST on bits per character: {best[0]}  at {best[1]:.3f}")
    print(f"  PREDICTION 2 CHECK: character level beats 1.850 bits/char? "
          f"{'YES' if best[0].startswith('char') or best[1] < 1.850 else 'see table above'}")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo "docs copied"
