"""diag_code_doctext.py — measure the CORPUS before spending a training run on it.

DOC_MODE="code" makes the training document the SOLUTION ONLY instead of problem + "\\n\\n" +
solution. Before training anything, check that it actually changed what is in the corpus.

The "both" numbers were measured earlier on the same sources (19,505,365 tokens, 43,254 documents):
    most common token   '`'     6.08%
    second              '<unk>' 5.74%   (1,119,649)
    third               'the'   5.55%
    43.0% of documents open with prose-heavy text
"""
import os
import re
import sys
from collections import Counter

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

PROSE_MARKERS = {"the", "is", "are", "you", "your", "task", "should", "function", "returns",
                 "given", "implement", "write", "solution", "example", "following", "input",
                 "output", "note", "must", "this", "that", "with", "for", "and", "of"}
CODE_MARKERS = {"(", ")", "[", "]", "{", "}", ":", "=", ".", ",", "def", "return", "if", "else",
                "for", "while", "import", "self", "class", "try", "except", "lambda"}
# the vocabulary the ORIGINAL corpus produced, for comparison
KNOWN_BOTH = {"backtick_pct": 6.08, "unk_pct": 5.74, "the_pct": 5.55, "prose_docs_pct": 43.0}

print("=" * 100)
print("MEASURING THE CODE-ONLY CORPUS — before training on it")
print("=" * 100)

import code_data                                       # noqa: E402
print(f"  DOC_MODE = {code_data.DOC_MODE!r}")
d = code_data.build()
vocab, ids, bounds, split = d["vocab"], d["ids"], d["bounds"], d["split"]
txt = [vocab[i] for i in ids]
n = len(txt)
print(f"  tokens {n:,}   documents {len(bounds)-1:,}   vocab {len(vocab):,}")
print()

cnt = Counter(txt)
prose = sum(c for t, c in cnt.items() if t.lower() in PROSE_MARKERS)
code = sum(c for t, c in cnt.items() if t in CODE_MARKERS)
bt = cnt.get("`", 0)
unk = cnt.get("<unk>", 0)
the = cnt.get("the", 0)

print(f"  {'measure':<34}{'BOTH (original)':>18}{'CODE (new)':>14}   verdict")
rows = [
    ("most common token is a backtick", KNOWN_BOTH["backtick_pct"], bt / n * 100),
    ("<unk> share", KNOWN_BOTH["unk_pct"], unk / n * 100),
    ("'the' share", KNOWN_BOTH["the_pct"], the / n * 100),
]
for name, b, c in rows:
    verdict = "IMPROVED" if c < b else ("same" if abs(c - b) < 0.2 else "WORSE")
    print(f"  {name:<34}{b:>17.2f}%{c:>13.2f}%   {verdict}")
print()
print(f"  prose-marker share  {prose/n*100:5.2f}%     code-marker share {code/n*100:5.2f}%")
print()
print("  THE TEN MOST COMMON TOKENS NOW:")
for t, c in cnt.most_common(10):
    lab = "PROSE" if t.lower() in PROSE_MARKERS else ("CODE" if t in CODE_MARKERS else "-")
    print(f"     {t!r:<18}{c:>10,}  ({c/n*100:5.2f}%)  {lab}")

# per-document opening composition
print()
print("  PER-DOCUMENT OPENING (first 40 tokens), sample of 200 val documents:")
spl = np.asarray(split)
nb, sampled = 0, 0
for j in range(len(bounds) - 1):
    if spl.ndim and spl[j] != 1:
        continue
    toks = [vocab[i] for i in ids[bounds[j]:bounds[j + 1]]]
    if len(toks) < 40:
        continue
    if sum(1 for t in toks[:40] if t.lower() in PROSE_MARKERS) >= 12:
        nb += 1
    sampled += 1
    if sampled >= 200:
        break
pct = nb / max(sampled, 1) * 100
print(f"     {nb}/{sampled} open with prose-heavy text ({pct:.1f}%)   "
      f"BOTH was {KNOWN_BOTH['prose_docs_pct']:.1f}%  -> "
      f"{'IMPROVED' if pct < KNOWN_BOTH['prose_docs_pct'] - 5 else 'no real change'}")
print()
print("  --- does the vocabulary now contain code identifiers? ---")
ident = [w for w in vocab if re.fullmatch(r"[a-z_][a-z_0-9]{2,}", w)
         and w not in PROSE_MARKERS and len(w) > 3]
print(f"     identifier-shaped tokens in vocab: {len(ident):,}  sample {ident[:14]}")
print()
print("=" * 100)
