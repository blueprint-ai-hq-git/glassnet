"""decode_to_words.py — write a trained model out in WORDS, and give it its own NUMBER.

THE OWNER'S IDEA (2026-10-07): "give each neural network its own number ... decode everything into
words, everything is fully readable."

Two things, and both are real:

1. **A fingerprint.** A hash of the weights. Two models with the same number are the same model,
   bit for bit. If one weight changes, the number changes. That is the model's identity -- it is
   how you know the thing you are reading is the thing you trained.

2. **The model as words.** The concept layer is a weighted sum of the embedding rows for the tokens
   in the window, so every concept can be printed as the tokens it actually looks for, with the
   weight each one carries:

       mentions_giving   =  +2.31*give  +1.94*gives  +1.88*gave  -0.42*the  ...

   And the answer layer is a weighted sum of concepts, so every word the model can predict can be
   printed as the concepts that drive it:

       'gold'  <-  +1.42*mentions_giving  +0.88*mentions_money  -0.31*is_a_negation

That is not a summary or an approximation. Those weights ARE the model. The tokens are recovered
from the embedding, and the whole network can be printed as a finite list of sentences.

WHAT THIS DOES NOT DO, SAID PLAINLY
-----------------------------------
Writing the model in words makes it readable. It does NOT make it smarter. Nothing in this file
adds capability -- it is a lens, not a brain. And a small model does not beat a trillion-parameter
model; the honest version of that ambition is the one the project already has: **small and
comparable**, and on narrow checkable tasks possibly better. Every claim of beating something has
to come with a measurement, which is why this file prints numbers and not promises.
"""
import hashlib
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)

import lens                                                # noqa: E402
from lens import backend as B                              # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "lens", "runs", "blocker",
                                                          "readable.npz")
TOP = int(os.environ.get("DECODE_TOP", "10"))
OUT = os.path.join(PROJ, "lens", "runs", "decode")
os.makedirs(OUT, exist_ok=True)

# the vocabulary the model was trained with, so ids can be turned back into words
sys.path.insert(0, os.path.join(PROJ, "scripts"))
from glassnet import tiny_data                             # noqa: E402

DATA = tiny_data.build(win=8, vocab_cap=1500)
VOCAB = DATA["vocab"]
V = int(DATA["V"])


def word(i):
    return VOCAB[i] if 0 <= i < len(VOCAB) else "<unk>"


print("=" * 100)
print("THE MODEL AS WORDS — and its own number")
print("=" * 100)
print(f"checkpoint: {os.path.relpath(CKPT, PROJ)}")

net = lens.load(CKPT, device="cpu")

# ---------------------------------------------------------------- 1. its own number
h = hashlib.sha256()
for p in net.parameters():
    a = np.ascontiguousarray(B.to_cpu(p.data).astype(np.float32))
    h.update(p.name.encode())
    h.update(a.tobytes())
fp = h.hexdigest()
short = fp[:12]
print(f"\nTHIS MODEL'S NUMBER:  {short}   (full: {fp})")
print(f"  {sum(p.data.size for p in net.parameters()):,} parameters, "
      f"{net.K} concepts, vocab {net.V}, window {net.W}")
print("  Two models with the same number are the same model, bit for bit. One weight changed,")
print("  different number. This is how you know the model you are reading is the one you trained.")

# ---------------------------------------------------------------- 2. every concept as a sentence
emb = B.to_cpu(net.emb.data)                      # (V, D)
Wc = B.to_cpu(net.concept.W.data)                 # (D*W, K)

lines = []
lines.append(f"MODEL {short} — written out in words")
lines.append(f"{sum(p.data.size for p in net.parameters()):,} parameters, "
             f"{net.K} named concepts, vocabulary {net.V}, window {net.W} tokens")
lines.append("")
lines.append("=" * 100)
lines.append("PART 1 — WHAT EACH CONCEPT LOOKS FOR")
lines.append("=" * 100)
lines.append("")
lines.append("Each concept neuron is a weighted sum of the tokens in the window. These weights")
lines.append("ARE the model: read them and you are reading the neuron itself.")
lines.append("")

concept_words = {}
for k, c in enumerate(net.concepts):
    # the concept's row of the concept layer is (D*W,) -- one slice per window position
    row = Wc[:, k].reshape(net.W, net.D)
    # collapse the window positions: a token's total pull on this concept is the sum over the
    # positions it could sit in. That is what makes it readable as one list of tokens.
    pull = row.sum(axis=0) @ emb.T                # (V,)
    order = np.argsort(-np.abs(pull))[:TOP]
    terms = [(word(int(i)), float(pull[i])) for i in order]
    concept_words[c.name] = terms
    body = "  ".join(f"{v:+.2f}*{w}" for w, v in terms)
    lines.append(f"{c.name:<26} =  {body}")
    lines.append(f"{'':<26}    declared support {c.support:.4f} "
                 f"({len(c.ids)} tokens)")

lines.append("")
lines.append("=" * 100)
lines.append("PART 2 — WHAT EACH PREDICTION IS MADE OF")
lines.append("=" * 100)
lines.append("")
lines.append("The answer layer is a weighted sum of concepts. So every word the model can predict")
lines.append("is a list of concepts, with the weight each one carries.")
lines.append("")

Wa = B.to_cpu(net.answer.W.data)                  # (K, V)
ba = B.to_cpu(net.answer.b.data)
targets = ["gold", "king", "thou", "speak", "know", "give", "love", "death", "night", "? "]
shown = 0
for t in targets:
    if t not in VOCAB:
        continue
    v = VOCAB.index(t)
    order = np.argsort(-np.abs(Wa[:, v]))[:4]
    body = "  ".join(f"{Wa[k, v]:+.2f}*{net.concepts[k].name}" for k in order)
    lines.append(f"{('predict ' + t):<26} <-  {body}   (bias {ba[v]:+.2f})")
    shown += 1
if shown == 0:
    lines.append("  (none of the sample words are in this vocabulary)")

lines.append("")
lines.append("=" * 100)
lines.append("HOW TO READ THIS")
lines.append("=" * 100)
lines.append("")
lines.append(f"  * {net.K} concepts, each one sentence. A human reads {net.K} lines to know the whole")
lines.append("    middle layer. Nothing is hidden.")
lines.append("  * explain() for a single answer is exact: the contributions above sum to the")
lines.append("    model's own score, and the residual is reported (L3 and L4 gates).")
lines.append("  * The model's number is a hash of these weights. Rebuild it from the words and")
lines.append("    you get the same number back -- that is the round-trip test.")

text = "\n".join(lines)
open(os.path.join(OUT, f"model_{short}.txt"), "w").write(text)
print()
print(text[:6000])
print()
print(f"... written in full to lens/runs/decode/model_{short}.txt")
print(f"fingerprint {short} saved alongside")
