"""optimise_diag.py — WHERE IS THIS MODEL WASTING CAPACITY? Measure, then fix.

The chat model run 1: 1.65M params, val ppl 353, next-token acc 11.08%, concepts 5/8.
It learned the FORMAT of dialogue and nothing else. Before changing anything, measure where the
capacity actually goes and where the information is destroyed.

THE SUSPECT, FROM THE SCALING RESULT
------------------------------------
The scaling test found the readable model's win was size-specific because **the concept count
stayed at 41 while the model grew 8x** -- the bottleneck came back. Here the concept layer is only
**NINE** numbers wide. So this measures the bottleneck ratio directly.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import chat_data                                           # noqa: E402

print("=" * 96)
print("OPTIMISATION DIAGNOSTIC — measure before changing anything")
print("=" * 96)

d = chat_data.build()
ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
V = len(vocab)
print(f"  corpus: {len(ids):,} tokens   vocab: {V:,}   conversations: {len(bounds):,}")

# ---------------------------------------------------------------- 1. how much is thrown away
print()
print("  1. INFORMATION THROWN AWAY AT THE INPUT")
print("  " + "-" * 92)
unk = float((ids == 0).mean())
print(f"     <unk> rate: {unk*100:.3f}% of all tokens")
print(f"     the <unk> row is {100.0/V:.4f}% of the embedding, and it carries no information")

# vocabulary coverage curve -- how many words do you actually need?
train_mask = np.zeros(len(ids), dtype=bool)
for j in range(len(bounds) - 1):
    if split[j] == 0:
        train_mask[bounds[j]:bounds[j + 1]] = True
tr = ids[train_mask]
cnt = Counter(tr.tolist())
total = sum(cnt.values())
run, need = 0, {}
for i, (tok, c) in enumerate(cnt.most_common()):
    run += c
    for frac in (0.90, 0.95, 0.99, 0.999):
        if frac not in need and run / total >= frac:
            need[frac] = i + 1
print(f"     words needed for 90.0% of tokens: {need.get(0.90, -1):,}")
print(f"     words needed for 95.0% of tokens: {need.get(0.95, -1):,}")
print(f"     words needed for 99.0% of tokens: {need.get(0.99, -1):,}")
print(f"     words needed for 99.9% of tokens: {need.get(0.999, -1):,}   (vocab is {V:,})")
singles = sum(1 for _, c in cnt.items() if c == 1)
print(f"     words appearing EXACTLY ONCE: {singles:,} "
      f"({singles/max(len(cnt),1)*100:.1f}% of the vocabulary)")

# ---------------------------------------------------------------- 2. where the parameters go
print()
print("  2. WHERE THE PARAMETERS GO  (d=128, window=8, K=9)")
print("  " + "-" * 92)
D, W, K = 128, 8, 9
emb = V * D
concept = D * W * K
answer = K * V
tot = emb + concept + answer
print(f"     embedding   V*D   = {V:,} * {D}      = {emb:>10,}   {emb/tot*100:5.1f}%")
print(f"     concept     D*W*K = {D*W} * {K}          = {concept:>10,}   {concept/tot*100:5.1f}%")
print(f"     answer      K*V   = {K} * {V:,}    = {answer:>10,}   {answer/tot*100:5.1f}%")
print(f"     TOTAL                                  = {tot:>10,}")

# ---------------------------------------------------------------- 3. THE BOTTLENECK
print()
print("  3. THE INFORMATION BOTTLENECK  <-- the suspect")
print("  " + "-" * 92)
inp = D * W
print(f"     the concept layer takes {inp:,} numbers in and puts {K} numbers out.")
print(f"     compression ratio: {inp/K:.0f} to 1")
print(f"     the answer layer must choose 1 of {V:,} words using only {K} numbers.")
print(f"     bits needed to name 1 of {V:,} words: {np.log2(V):.1f}")
print(f"     so {K} numbers must carry {np.log2(V):.1f} bits = {np.log2(V)/K:.2f} bits each,")
print(f"     AND they must ALSO carry the meaning needed to predict correctly.")
print()
print("     the scaling experiment measured where this breaks:")
print("       9 concepts  -> gap +23.045   (FAIL)")
print("      22 concepts  -> gap +20.203   (FAIL)")
print("      28 concepts  -> gap +19.937   (FAIL)")
print("      34 concepts  -> gap +16.033   (FAIL)")
print("      41 concepts  -> gap  +0.342   (WIN)")
print(f"     at d=256 with window 8, the input is {256*8:,} numbers, and 41 was enough.")
print(f"     HERE the input is {inp:,} numbers and K is only {K}. Ratio {inp/K:.0f}:1 vs {256*8/41:.0f}:1 at the win.")
print()
print(f"     >>> PREDICTION: widening K is the single highest-value change. The layer is")
print(f"     >>> {K} wide where the measured win needed 41 for a 4x LARGER input.")

# ---------------------------------------------------------------- 4. other levers, ranked
print()
print("  4. OTHER LEVERS, RANKED BY EXPECTED VALUE")
print("  " + "-" * 92)
levers = [
    ("widen the concept layer (K)", "HIGH", "measured directly by the scaling experiment"),
    ("learning-rate decay", "MEDIUM", "lr is constant 0.4; decay usually helps late training"),
    ("smaller vocab / better tokenisation", f"MEDIUM", f"{unk*100:.2f}% <unk>; {singles:,} singletons"),
    ("longer training", "MEDIUM", "ppl was still falling at epoch 3 (523->400->353)"),
    ("bigger d", "LOW-MED", "the bottleneck is K, not d; d=384 with K=9 is a worse ratio"),
    ("window size", "LOW", "8 tokens is short but changing it changes the task"),
]
for name, val, why in levers:
    print(f"     {val:<8} {name:<34} {why}")
print()
print("  NOTE ON THE QUEUED RUN: d=384 with K=9 makes the bottleneck RATIO WORSE")
print(f"  ({384*8} in / 9 out = {384*8/9:.0f}:1). So a wider-concept run is needed alongside it,")
print("  and the two together isolate which change did the work.")
print("=" * 96)
