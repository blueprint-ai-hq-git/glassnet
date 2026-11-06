"""diag_unk_targets.py — why <unk> survives a 5x bigger vocabulary.

The reasoning that was WRONG:
  "<unk> is 7.21% of the corpus, and it sits where identifiers go, so a bigger vocabulary will let
   the model name things and executability will improve."
A 32k vocabulary (5x the parameters) gave: ppl 39.05 -> 54.30, executability 6/25 -> 6/25.

The hypothesis to test now:
  **The model is not failing to KNOW the words. It is CHOOSING `<unk>` as an output, because `<unk>`
  is one of the most frequent tokens in the training targets. A next-token model trained on text
  containing `<unk>` learns that `<unk>` is a good continuation, and enlarging the vocabulary does
  not change that at all -- it only changes how often the INPUT is unknown.**

Measure: what fraction of the tokens the model is asked to PREDICT are `<unk>`? If that number is
large, the model is being trained to emit it, and the fix is to stop rewarding it.
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

os.environ.setdefault("CODE_DOC", "code")
import code_data                                        # noqa: E402

print("=" * 100)
print("WHY <unk> SURVIVES A BIGGER VOCABULARY")
print("=" * 100)

for vocab_cap in (6000, 32000):
    os.environ["CD_VOCAB"] = str(vocab_cap)
    import importlib
    importlib.reload(code_data)
    d = code_data.build()
    vocab, ids, bounds, split = d["vocab"], d["ids"], d["bounds"], d["split"]
    W = 16
    Xtr, Ytr = code_data.make_windows(ids, bounds, split, W, which=0)
    Y = np.asarray(Ytr).reshape(-1)
    unk_id = 0
    frac_target = float((Y == unk_id).mean())
    frac_corpus = float((np.asarray(ids) == unk_id).mean())
    print(f"  vocab {vocab_cap:,}")
    print(f"    <unk> share of the CORPUS        : {frac_corpus*100:6.2f}%")
    print(f"    <unk> share of the TRAINING TARGETS: {frac_target*100:6.2f}%   <- what it is trained to emit")
    print(f"    training windows {len(Y):,}")
    print(f"    of {len(Y):,} targets, {int((Y == unk_id).sum()):,} are `<unk>`")
    # what is the most frequent TARGET, and how does <unk> rank?
    cnt = np.bincount(Y, minlength=len(vocab))
    order = np.argsort(-cnt)
    print("    the five most frequent TARGETS:")
    for r, k in enumerate(order[:5]):
        print(f"      {r+1}. {vocab[k]!r:<16} {cnt[k]:>10,}  ({cnt[k]/len(Y)*100:5.2f}%)")
    print()
    if vocab_cap == 32000:
        print("  => <unk> is still among the most frequent TARGETS. The model is rewarded for")
        print("     emitting it thousands of times, so it does. A bigger vocabulary changes the")
        print("     INPUT, not the reward.")
        print()
        print("  THE FIX: MASK the loss at positions whose target is `<unk>`.")
        print("     Those positions are unanswerable by construction -- the correct token was")
        print("     thrown away by the vocabulary. Training on them teaches the model to produce")
        print("     `<unk>`, which is exactly the behaviour that makes the output unusable.")
    print()
print("=" * 100)
