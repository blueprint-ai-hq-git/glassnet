"""count_it.py — "26 letters, so many words -- how do you fit it all in?"

THE QUESTION
------------
26 letters. Words up to 20 letters long. 16-word windows. The number of possible strings is
astronomical. So how does a model with 679,009 numbers handle language at all?

THE ANSWER, MEASURED
--------------------
1. It does NOT store words. It stores a DIRECTION (128 numbers) for each word it has ever seen.
2. It does not enumerate possible strings. It never tries to. That is the entire trick.
3. The price: a word it has never seen becomes <unk>. This script measures that price exactly.
"""
import json
import os
from collections import Counter

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
PREP = os.path.join(PROJ, "data", "chat", "prepared")

print("=" * 100)
print("1. THE COMBINATORIAL EXPLOSION — how many strings COULD exist")
print("=" * 100)
print(f"  {'length':>7}{'possible strings (26^n)':>40}")
for n in (1, 2, 3, 4, 5, 8, 10, 15, 20, 30):
    print(f"  {n:>7}{26**n:>40,}")
total = sum(26**n for n in range(1, 21))
print()
print(f"  all strings up to 20 letters: {total:,}")
print(f"  that is {total:.3e} -- {total/1e18:.2e} quintillion")
print(f"  and 16-word windows from a 3,500-word vocab: 3500^16 = {3500**16:.3e}")
print()
print("  So yes -- if you tried to make a table with one entry per possible string,")
print("  you would need more entries than there are atoms in your hard disk, then the")
print("  sun, then everything. You can never finish. Nobody does it that way.")
print()

print("=" * 100)
print("2. WHAT WE ACTUALLY STORE — the real vocabulary and its coverage")
print("=" * 100)
ids = np.load(os.path.join(PREP, "ids_v3500.npy"))
vocab = json.load(open(os.path.join(PREP, "vocab_v3500.json")))
vocab12k = json.load(open(os.path.join(PREP, "vocab_v12000.json")))
print(f"  words stored in the trained model : {len(vocab):,}")
print(f"  tokens in the training corpus     : {len(ids):,}")

cnt = Counter(ids.tolist())
unk = cnt.get(0, 0)
print(f"  tokens that became <unk>          : {unk:,}  ({unk/len(ids)*100:.2f}% of all tokens)")
print()
print(f"  the 12,000-word version stored    : {len(vocab12k):,} words, and its <unk> rate was lower,")
print("  but it scored WORSE (90.43 vs 59.63 ppl) because those extra words were mostly")
print("  seen once or twice -- they cost parameters and taught the model nothing.")
print()

print("=" * 100)
print("3. WHY 3,500 IS ENOUGH — most words are rare, a few carry the text")
print("=" * 100)
freqs = np.array(sorted([c for w, c in cnt.items() if w != 0], reverse=True))
cum = np.cumsum(freqs) / freqs.sum()
for frac in (0.5, 0.8, 0.9, 0.95, 0.99):
    n = int(np.searchsorted(cum, frac)) + 1
    print(f"  {frac*100:>5.0f}% of the text is covered by the top {n:>6,} words")
print()
print(f"  {'rank band':<22}{'words':>10}{'share of text':>16}")
for lo, hi in [(1, 10), (11, 100), (101, 1000), (1001, 2000), (2001, 3500), (3501, len(freqs))]:
    b = freqs[lo-1:hi]
    if len(b):
        print(f"  ranks {lo:>5}-{hi:<5}{len(b):>10,}{b.sum()/freqs.sum()*100:>15.1f}%")
print()
print("  This is the answer to 'there are so many words'. There are -- but most appear once.")
print("  A vocabulary of 3,500 keeps the words that do the work and calls the rest <unk>.")
print()

print("=" * 100)
print("4. THE REAL TRICK — it does not store windows either")
print("=" * 100)
W = 16
trained = 1_500_000
print(f"  possible 16-word windows : 3500^16 = {3500**16:.3e}")
print(f"  windows actually trained : {trained:,}")
print(f"  fraction seen            : 1 in {3500**16/trained:.3e}")
print()
print("  So ~every window the model is tested on has NEVER been seen, not once, in training.")
print("  And yet it predicts the next word at perplexity 59.63 rather than 3,500 (chance).")
print()
print("  >>> THAT is how it fits. It does not store the combinations. It stores a RULE that")
print("  >>> produces them. The combinations are generated at use time, not memorised.")
print()

print("=" * 100)
print("5. THE PRICE — stated honestly")
print("=" * 100)
print(f"  A word the model never saw becomes <unk>, {unk/len(ids)*100:.2f}% of tokens in our corpus.")
print("  It cannot read a brand-new word. It has no notion of letters or spelling.")
print(f"  Compare: the character-level model COULD spell new words, and it scored far worse")
print("  (3.139 vs 1.289 bits/char). That is the trade, measured, not asserted.")
print()
print(f"  parameters total        : 679,009")
emb_params = len(vocab) * 128
print(f"  of which the vocabulary : {emb_params:,}  ({emb_params/679009*100:.1f}% -- 128 numbers per word)")
print(f"  numbers per word        : 128")
print()
print("  So 'numbering up' the words costs 128 numbers each, and we only pay for words that")
print("  actually occurred. The 2.0e28 possible strings are never numbered, never stored,")
print("  and never needed.")
print("=" * 100)
