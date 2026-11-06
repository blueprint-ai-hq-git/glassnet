"""how_numbers.py — HOW A WORD GETS ITS NUMBER. The actual arithmetic, opened up.

The claim under test: `sorry(+258.40)` for the concept `is_an_apology`.

Where does 258.40 come from? Not from a list anywhere. It is a DOT PRODUCT between two sets of
numbers the model learned:

    the concept's own weight vector   (what "apology" means to this model, as 128 numbers)
    the word's embedding vector       (what "sorry" means to this model, as 128 numbers)

The concept weights are shaped (window, D). They are summed over the window positions to give ONE
D-vector, then dotted with every word's embedding at once. That produces one number per word in the
vocabulary -- 3,500 numbers -- and the top ones are simply the largest by absolute value.

This script prints every step for real, including the elementwise products, so the number can be
checked by hand.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import lens
from lens import backend as B

P = os.path.join(PROJ, "lens", "runs", "chat")
PREP = os.path.join(PROJ, "data", "chat", "prepared")

net = lens.load(os.path.join(P, "f1_long_ep60.npz"), device="cpu")
vocab = json.load(open(os.path.join(PREP, "vocab_v3500.json")))
vmap = {w: i for i, w in enumerate(vocab)}

emb = B.to_cpu(net.emb.data)          # (V+1, D) -- one row of D numbers per word
Wc = B.to_cpu(net.concept.W.data)     # (D*W, K) -- each column is one concept's weights
W, D, K = net.W, net.D, net.K

print("=" * 100)
print("THE SHAPES — what is stored, exactly")
print("=" * 100)
print(f"  embedding matrix emb : {emb.shape}   = {emb.shape[0]:,} words x {emb.shape[1]} numbers each")
print(f"  concept weights Wc   : {Wc.shape}   = {Wc.shape[0]} numbers x {Wc.shape[1]} concepts")
print(f"  so the WHOLE model's meaning lives in {emb.size + Wc.size:,} numbers, all inspectable.")
print()

k = next(i for i, c in enumerate(net.concepts) if c.name == "is_an_apology")
name = net.concepts[k].name
print("=" * 100)
print(f"STEP 1 — the concept '{name}' is a column of numbers, reshaped to ({W}, {D})")
print("=" * 100)
row = Wc[:, k].reshape(W, D)
print(f"  raw column shape {Wc[:, k].shape} -> reshaped {row.shape}")
print(f"  first 3 window positions, first 8 of {D} numbers each:")
for i in range(3):
    print(f"    pos {i}: [{', '.join(f'{v:+.4f}' for v in row[i, :8])} ...]")
print()
print("  Why (window, D)? Each of the 16 word-positions in the window gets its OWN weight vector,")
print("  because 'sorry' at position 3 and 'sorry' at position 12 should not mean the same thing.")
print()

print("=" * 100)
print("STEP 2 — sum over the 16 positions -> ONE vector of 128 numbers")
print("=" * 100)
cw = row.sum(axis=0)
print(f"  concept vector cw = row.sum(axis=0)   shape {cw.shape}")
print(f"  first 8 of 128: [{', '.join(f'{v:+.4f}' for v in cw[:8])} ...]")
print()

print("=" * 100)
print("STEP 3 — the word 'sorry' is ALSO 128 numbers")
print("=" * 100)
wi = vmap["sorry"]
wv = emb[wi]
print(f"  'sorry' is vocabulary id {wi}")
print(f"  its embedding: [{', '.join(f'{v:+.4f}' for v in wv[:8])} ...]")
print()

print("=" * 100)
print("STEP 4 — THE COMBINATION: multiply element by element, then add them all up")
print("=" * 100)
prod = cw * wv
print(f"  {'index':>6}{'concept cw':>14}{'word wv':>14}{'product':>14}")
for i in range(8):
    print(f"  {i:>6}{cw[i]:>14.4f}{wv[i]:>14.4f}{prod[i]:>14.4f}")
print(f"  {'...':>6}{'...':>14}{'...':>14}{'...':>14}")
print()
print(f"  sum of ALL {D} products = {prod.sum():+.4f}")
print()
print(f"  >>> the reported weight for 'sorry' was +258.40")
print(f"  >>> computed here, from the raw numbers, it is {prod.sum():+.4f}")
print(f"  >>> MATCH: {abs(prod.sum() - 258.40) < 0.01}")

print()
print("=" * 100)
print("STEP 5 — do that for ALL 3,500 words, then SELECT the biggest")
print("=" * 100)
pull = cw @ emb.T                      # one dot product per word -- this IS the whole "word list"
print(f"  pull = cw @ emb.T   ->  shape {pull.shape}  = one number per word")
print()
order = np.argsort(-np.abs(pull))      # selection = sort by absolute value, biggest first
print(f"  order = np.argsort(-np.abs(pull))   <- THIS is how the words are selected")
print()
print(f"  {'rank':>5}{'word':>16}{'number':>12}   {'|number|':>10}")
for r in range(12):
    i = order[r]
    print(f"  {r+1:>5}{vocab[i]:>16}{pull[i]:>12.2f}   {abs(pull[i]):>10.2f}")
print()
tot = np.abs(pull).sum()
top40 = np.abs(pull[order[:40]]).sum()
print(f"  sum of |all 3,500 numbers|      = {tot:.2f}")
print(f"  sum of |the top 40 numbers|     = {top40:.2f}")
print(f"  coverage = {top40:.2f} / {tot:.2f} = {top40/tot*100:.1f}%")
print()
print("  >>> THAT is where 'the top 40 words explain 4.8% of this neuron' comes from.")
print("  >>> The other 95.2% is spread across the remaining 3,460 words, each too small to notice.")

print()
print("=" * 100)
print("SO: HOW ARE THE NUMBERS DETERMINED?")
print("=" * 100)
print("  1. Each word starts as 128 random numbers and is pushed around by training.")
print("  2. Each concept is also 16x128 numbers, pushed around by training.")
print("  3. The number you see for a word is the DOT PRODUCT of the concept vector and that")
print("     word's vector -- a similarity score between two learned directions.")
print("  4. Words are SELECTED by sorting those 3,500 numbers by absolute value.")
print()
print("  Nothing is hand-assigned. No word list was written by a human. The numbers come out of")
print("  gradient descent, and this script recomputes one of them from the raw stored numbers.")
print("=" * 100)
