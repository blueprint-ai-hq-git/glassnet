"""show_task.py — THE EXACT TASK, THE EXACT DATA, THE EXACT WORDS.

Answers with artifacts, not description:
  1. What task?      -> the literal training data and the literal objective
  2. What words?     -> the concept word lists with exact weights, printed
  3. Transparent?    -> explain() output, including its own stated limit
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import lens
from lens.concepts import concept_word_list, explain_text

P = os.path.join(PROJ, "lens", "runs", "chat")
PREP = os.path.join(PROJ, "data", "chat", "prepared")

log = json.load(open(os.path.join(P, "f1_long_log.json")))
a = log["args"]
VOCAB_CAP = a.get("vocab_cap", 3500)

print("=" * 100)
print("1. THE EXACT TRAINING TASK")
print("=" * 100)
print(f"  objective      : predict the NEXT WORD given the previous {a['window']} words")
print(f"  loss           : cross-entropy on that next word, plus a concept auxiliary loss")
print(f"  window         : {a['window']} words of context")
print(f"  vocabulary     : {VOCAB_CAP:,} words, built from TRAIN text ONLY")
print(f"  batch / lr     : {a['batch']} / {a['lr']}, linearly decayed to {a['lr']*(1-a['lr_decay']):.2f}")
print(f"  epochs         : {a['epochs']}    device: {a['device']}")
print()
try:
    ids = np.load(os.path.join(PREP, f"ids_v{VOCAB_CAP}.npy"))
    vocab = json.load(open(os.path.join(PREP, f"vocab_v{VOCAB_CAP}.json")))
    W = a["window"]
    print("  THE ACTUAL DATA — three real training windows, verbatim:")
    for s in (20000, 400000, 3000000):
        ctx = " ".join(vocab[i] for i in ids[s:s + W])
        nxt = vocab[ids[s + W]]
        print(f"\n    input  = {ctx}")
        print(f"    answer = {nxt!r}")
    print()
    print("  That is the whole task. Nothing else was trained on.")
except Exception as e:
    print(f"  (could not load raw ids: {type(e).__name__}: {e})")

ck = os.path.join(P, "f1_long_ep60.npz")
net = lens.load(ck, device="cpu")
vocab = json.load(open(os.path.join(PREP, f"vocab_v{VOCAB_CAP}.json")))
print(f"\n  loaded {os.path.basename(ck)}   concepts: {len(net.concepts)}   "
      f"params: {sum(p.data.size for p in net.parameters()):,}")

print()
print("=" * 100)
print("2. WHAT WORDS COME OUT — the concept word lists, with exact weights")
print("=" * 100)
print("  Each line = the words a concept neuron pulls on, and HOW HARD (weight).")
print()
for k, c in enumerate(net.concepts[:12]):
    terms, frac = concept_word_list(net, k, top=12)
    shown = ", ".join(f"{vocab[t]}({w:+.2f})" for t, w in terms[:12])
    print(f"  {c.name:<24} {shown}")
    print(f"  {'':<24} top-12 words explain {frac*100:.1f}% of this neuron")

print()
print("=" * 100)
print("3. FULL TRANSPARENCY — explain() printing its OWN limit")
print("=" * 100)
try:
    W_ = a["window"]
    win = ids[20000:20000 + W_]
    print(explain_text(net, win, vocab=vocab, top_k=6))
except Exception as e:
    print(f"  explain_text error: {type(e).__name__}: {e}")

print()
print("  --- how much of EACH concept neuron its top-40 words explain ---")
fracs = []
for k, c in enumerate(net.concepts):
    _, f = concept_word_list(net, k, top=40)
    fracs.append((c.name, f))
lo = min(f for _, f in fracs)
hi = max(f for _, f in fracs)
print(f"    range: {lo*100:.1f}%  to  {hi*100:.1f}%   (median {np.median([f for _,f in fracs])*100:.1f}%)")
print(f"    -> so a word list shows a SLICE. The model prints this itself rather than hiding it.")
print()
print("    best and worst described concepts:")
for n, f in sorted(fracs, key=lambda x: -x[1])[:4]:
    print(f"      {n:<24} {f*100:5.1f}%  {'#'*int(f*50)}")
print("      ...")
for n, f in sorted(fracs, key=lambda x: x[1])[:4]:
    print(f"      {n:<24} {f*100:5.1f}%  {'#'*int(f*50)}")
print("=" * 100)
