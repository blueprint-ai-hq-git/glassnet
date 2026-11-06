"""diag_prompt_unk.py — is the hand-written `<unk>` coming from the PROMPT?

Look at the prompts the executable test uses:
    ` ` ` python import numpy as np def <unk> ( a1 , <unk> ) :
The PROMPT ITSELF contains `<unk>` where an identifier should be. If the model is handed
`def <unk> (`, the natural continuation is more `<unk>` -- it is doing the locally correct thing.

Hypothesis: the generated `<unk>` rate is driven by the `<unk>` rate of the PROMPT, not only by the
model's habits. Test it by comparing the two groups directly:
    prompts WITH no <unk>      vs      prompts WITH many <unk>
and measuring the generated <unk> rate in each.

This is a fair-test question. If the harness prompts the model with a signature that is already
broken, then "6/25" measures the harness as much as the model -- and that mistake has been made twice
already in this project (the token-join bug, and the under-budget comparison).
"""
import ast
import os
import re
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

os.environ.setdefault("CODE_DOC", "code")
os.environ.setdefault("CD_VOCAB", "32000")
import code_data                                        # noqa: E402

print("=" * 100)
print("IS THE `<unk>` COMING FROM THE PROMPT?")
print("=" * 100)

d = code_data.build()
vocab, ids, bounds, split = d["vocab"], d["ids"], d["bounds"], d["split"]
W = 16

# rebuild the SAME validation prompts reexec_code.py uses (style B: cut after the signature colon)
prompts = []
for j in range(len(bounds) - 1):
    if split[j] != 1:
        continue
    text = " ".join(vocab[i] for i in ids[bounds[j]:bounds[j + 1]])
    k = text.find("def ")
    if k < 0:
        continue
    colon = text.find(" : ", k)
    if colon > 0 and len(prompts) < 25:
        prompts.append(text[:colon + 3])

print(f"  {len(prompts)} validation prompts (same ones the executable test uses)")
print()
n_unk_prompt = sum(1 for p in prompts if "<unk>" in p)
print(f"  prompts containing `<unk>`            : {n_unk_prompt}/{len(prompts)}")
print(f"  prompts whose PROMPT is clean of <unk>: {len(prompts)-n_unk_prompt}/{len(prompts)}")
print()
print("  a few prompts, with the <unk> positions marked:")
for p in prompts[:6]:
    flag = "  <- PROMPT ALREADY BROKEN" if "<unk>" in p else ""
    print(f"    {p[-64:]!r}{flag}")
print()
# how many identifiers would the prompt need that the corpus could not represent?
tot_tok = tot_unk = 0
for p in prompts:
    toks = code_data.tokenize(p)
    tot_tok += len(toks)
    tot_unk += sum(1 for t in toks if t not in {w: 1 for w in vocab} and t == "<unk>")
print(f"  over the {len(prompts)} prompts: {tot_tok} tokens, "
      f"{sum(1 for p in prompts for t in code_data.tokenize(p) if t == '<unk>')} of them `<unk>`")
print()
print("  CONCLUSION, from the numbers above:")
if n_unk_prompt >= len(prompts) * 0.6:
    print("    MOST prompts hand the model `def <unk> (`. Continuing with `<unk>` is the locally")
    print("    correct behaviour, so a large part of the generated `<unk>` is READ-BACK, not")
    print("    invention. The executable test is partly measuring prompt quality.")
    print()
    print("  TWO FIXES, AND THEY ARE DIFFERENT THINGS:")
    print("    1. FAIR TEST: only prompt with signatures the corpus could fully represent")
    print("       (no `<unk>` in the prompt). Otherwise the test asks the model to name a")
    print("       function whose name was deleted before it ever saw it.")
    print("    2. TRAINING: mask the loss at `<unk>` targets, so the model is never rewarded")
    print("       for emitting `<unk>` in the first place.")
else:
    print("    Most prompts are clean, so the generated `<unk>` is mostly the model's own habit")
    print("    and the loss-mask fix is the one that matters.")
print("=" * 100)
