"""measure_vocab_need.py — how big does the code vocabulary have to be?

The measured problem: `<unk>` is the MOST COMMON token in the code-only corpus at 7.21%. `<unk>` is
where identifiers go, so a model with this vocabulary CANNOT NAME a function -- which is exactly what
the execution test showed (`def <unk> (` over and over).

Before building anything, measure the requirement: what vocabulary size brings `<unk>` down to a
level where identifiers are expressible?

Also count what is being EVICTED, which matters more than the raw rate: if the evicted tokens are
names, the model cannot write `def two_sum(nums, target)`, only `def <unk> ( <unk> , <unk> )`.
"""
import os
import sys
from collections import Counter

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

os.environ.setdefault("CODE_DOC", "code")
import code_data                                        # noqa: E402

print("=" * 100)
print("HOW BIG DOES THE CODE VOCABULARY NEED TO BE?")
print("=" * 100)
print(f"  DOC_MODE = {code_data.DOC_MODE}")

pairs = list(code_data.python_pairs(limit=60000))
print(f"  {len(pairs):,} python pairs")
print()

# token frequency over the code-only documents (same text the model would see)
freq = Counter()
for p, s in pairs:
    freq.update(code_data.tokenize(code_data.doc_text(p, s).lower()))
total = sum(freq.values())
print(f"  {total:,} tokens   {len(freq):,} distinct tokens")
print()

print("  --- what a vocabulary of each size actually keeps ---")
print(f"  {'vocab':>8}{'<unk> rate':>13}{'tokens kept':>14}{'#unk tokens':>14}   example of what is LOST")
import re
IDENT = re.compile(r"[a-z_][a-z_0-9]{2,}")
for cap in (6000, 12000, 20000, 32000, 50000):
    kept = set(w for w, _ in freq.most_common(cap - 1))
    unk = total - sum(c for w, c in freq.items() if w in kept)
    lost_ids = [(w, c) for w, c in freq.most_common() if w not in kept and IDENT.fullmatch(w)]
    ex = ", ".join(w for w, _ in lost_ids[:4])
    print(f"  {cap:>8,}{unk/total*100:>12.2f}%{total-unk:>14,}{unk:>14,}   {ex}")

print()
print("  --- the decisive question: where does <unk> land in real solutions? ---")
# take a few solutions, encode them at 6000, and show how many identifiers become <unk>
v6000 = {w: i for i, w in enumerate(["<unk>"] + [w for w, _ in freq.most_common(5999)])}
v32000 = {w: i for i, w in enumerate(["<unk>"] + [w for w, _ in freq.most_common(31999)])}
shown = 0
for p, s in pairs:
    if "def " not in s or len(s) > 700:
        continue
    toks = code_data.tokenize(s.lower())
    if not (300 <= len(toks) <= 500):
        continue
    u6 = sum(1 for t in toks if t not in v6000)
    u32 = sum(1 for t in toks if t not in v32000)
    print(f"    a {len(toks):>4}-token solution: <unk> at 6k = {u6:>4} ({u6/len(toks)*100:5.1f}%)   "
          f"at 32k = {u32:>4} ({u32/len(toks)*100:5.1f}%)")
    shown += 1
    if shown >= 5:
        break

print()
print("=" * 100)
print("  THE NUMBER THIS PRODUCES IS THE VOCABULARY SIZE TO USE. Nothing is built yet.")
print("=" * 100)
