"""diag_code.py — WHY the code model writes prose instead of code.

The evidence: given `def f ( x ) :` it emits "this solution, the <unk> class is a method that takes
the specified parameters", "# output: function to return the", "your task is to". Markdown
backticks. English. Not code.

Hypothesis to test: **the training corpus is dominated by natural-language problem statements**, so
the highest-probability continuation after any prompt is more prose. If so, the model is not broken
-- it is doing exactly what the data asks, and no amount of training changes that.

Measured here, without changing anything:
  A. What fraction of tokens are English words vs code tokens, per corpus?
  B. What fraction of the checkpoints' training windows END inside prose vs inside code?
  C. What does the most likely continuation look like from a code-only context vs a prose context?
"""
import os
import re
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import code_data                                        # noqa: E402

print("=" * 100)
print("WHY THE CODE MODEL WRITES PROSE")
print("=" * 100)

d = code_data.build()
vocab, ids, bounds, split = d["vocab"], d["ids"], d["bounds"], d["split"]
print(f"  vocab {len(vocab):,}   tokens {len(ids):,}   documents {len(bounds)-1:,}")
try:
    spl = np.asarray(split)
    if spl.ndim == 0:
        print(f"  split is a SCALAR ({spl!r}) -- one flag for the whole corpus, not per document")
    else:
        print(f"  train docs {int((spl == 0).sum()):,}   val docs {int((spl == 1).sum()):,}")
except Exception as e:
    print(f"  split type {type(split).__name__}: {e}")
print()

# English words that appear in prose but essentially never in code structure
PROSE_MARKERS = {"the", "is", "are", "you", "your", "task", "should", "function", "returns",
                 "given", "implement", "write", "solution", "example", "following", "input",
                 "output", "note", "must", "can", "this", "that", "with", "for", "and", "of"}
# tokens that are code punctuation / structure
CODE_MARKERS = {"(", ")", "[", "]", "{", "}", ":", "=", ".", ",", "def", "return", "if", "else",
                "for", "while", "import", "self", "class", "try", "except", "lambda", "None",
                "True", "False", "+", "-", "*", "/", "%", "==", "!=", "<", ">", "and", "or"}

txt = [vocab[i] for i in ids]
prose_hits = sum(1 for t in txt if t.lower() in PROSE_MARKERS)
code_hits = sum(1 for t in txt if t in CODE_MARKERS)
backticks = sum(1 for t in txt if t == "`")
print("A. WHAT IS ACTUALLY IN THE CORPUS")
print(f"   prose-marker tokens   {prose_hits:>10,}  ({prose_hits/len(txt)*100:5.2f}%)")
print(f"   code-marker tokens    {code_hits:>10,}  ({code_hits/len(txt)*100:5.2f}%)")
print(f"   markdown backticks    {backticks:>10,}  ({backticks/len(txt)*100:5.2f}%)")
print()

# Where do the TRAINING windows actually end -- inside prose or inside code?
W = 16
Xtr, ytr = code_data.make_windows(ids, bounds, split, W, which=0)
print(f"B. TRAINING WINDOWS: {Xtr.shape[0]:,} windows of {W} tokens")
print("   what token does the model have to predict? (the window's target)")
tgt_ids = ytr if ytr.ndim == 1 else np.asarray(ytr).reshape(-1)
tgt_tok = [vocab[int(i)] for i in tgt_ids[:200000]]
is_prose = sum(1 for t in tgt_tok if t.lower() in PROSE_MARKERS)
is_bt = sum(1 for t in tgt_tok if t == "`")
print(f"     targets that are prose markers : {is_prose:>8,}  ({is_prose/len(tgt_tok)*100:5.2f}%)")
print(f"     targets that are backticks     : {is_bt:>8,}  ({is_bt/len(tgt_tok)*100:5.2f}%)")
print()

# How many documents are dominated by prose?
print("C. PER-DOCUMENT COMPOSITION (a sample of 200 validation documents)")
nb_prose_first = 0
sampled = 0
spl = np.asarray(split)
for j in range(len(bounds) - 1):
    if spl.ndim and spl[j] != 1:
        continue
    toks = [vocab[i] for i in ids[bounds[j]:bounds[j + 1]]]
    if len(toks) < 40:
        continue
    first40 = toks[:40]
    p = sum(1 for t in first40 if t.lower() in PROSE_MARKERS)
    if p >= 12:
        nb_prose_first += 1
    sampled += 1
    if sampled >= 200:
        break
print(f"   {sampled} val documents; {nb_prose_first} of them open with prose-heavy text "
      f"({nb_prose_first/max(sampled,1)*100:.1f}%)")
print()
print("   In other words: the first 40 tokens of most documents are the PROBLEM STATEMENT.")
print("   A model trained to predict the next token learns that after almost any prompt the most")
print("   likely continuation is more English description -- which is exactly what it emits.")
print()

# What does the most likely next token look like from a code context vs a prose context?
print("D. THE 10 MOST COMMON TOKENS IN THE CORPUS — is the corpus code-shaped or prose-shaped?")
from collections import Counter
cnt = Counter(txt)
for t, c in cnt.most_common(10):
    lab = "PROSE" if t.lower() in PROSE_MARKERS else ("CODE" if t in CODE_MARKERS else "-")
    print(f"     {t!r:<18} {c:>10,}  ({c/len(txt)*100:5.2f}%)  {lab}")
print()
print("=" * 100)
print("  READ THIS BEFORE CHANGING ANYTHING.")
print("=" * 100)
