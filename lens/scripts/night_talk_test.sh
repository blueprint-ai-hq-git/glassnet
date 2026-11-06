#!/bin/bash
# NIGHT TALK TEST - stop the heavy work, just ask the model a few things and show exactly what it says.
# Nothing cherry-picked: every answer is printed verbatim, plus the ship filename and its measured honest
# accuracy, so the text and the number are seen together.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest"

SHIP="${1:-$(ls -t lens/runs/chat/ld_d1024_ctrl_24_ep24.npz 2>/dev/null | head -1)}"
[ -f "$SHIP" ] || SHIP=$(ls -t lens/runs/chat/*.npz | head -1)

.venv-gpu/bin/python3 -u - "$SHIP" <<'PY'
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "lens"); sys.path.insert(0, "lens/scripts")
import lens, chat_data
from lens.decode import generate

ship = sys.argv[1]
net = lens.load(ship, device="cpu")
p = sum(x.data.size for x in net.parameters())
d = chat_data.build(); vocab = list(d["vocab"])
# match the ship: control-token models carry two extra ids (12000 <|im_start|>, 12001 <|im_end|>)
if net.V == len(vocab) + 2:
    vocab = vocab + ["<|im_start|>", "<|im_end|>"]
    print(f"  (this ship uses the standard chat control tokens, vocab {net.V})")
elif net.V != len(vocab):
    raise SystemExit(f"vocab mismatch: ship has {net.V}, word list has {len(vocab)}")
vmap = {w: i for i, w in enumerate(vocab)}
print(f"MODEL: {os.path.basename(ship)}")
print(f"  parameters {p:,} ({p/1e6:.1f}M)   window {net.W}   concepts {net.K}")
print(f"  measured honest accuracy for this run is in lens/runs/LADDER-CTRL.txt next to the rung name")
print()
questions = ["hello , how are you ?",
             "what is 2 + 2 ?",
             "who are you ?",
             "tell me about the sun .",
             "what is water made of ?"]
for q in questions:
    g = generate(net, vocab, vmap, f"human : {q} assistant :", max_new=24, rule_stop=True)
    txt = g["text"].strip().replace("\n", " ")
    print(f"  Q: {q}")
    print(f"  A: {txt}")
    print(f"     (stop reason: {g['stop_reason']}, {g['tokens']} tokens)")
    print()
print("READING: this is the raw output. If it is word salad, the honest verdicts in LADDER-CTRL.txt say so")
print("with measured numbers; the text above is what the model actually produces, nothing selected.")
PY
