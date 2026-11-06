"""chat_concepts.py — the named concepts for a CONVERSATIONAL model.

The architecture requires named concepts, so they have to be things that mean something in a
conversation. Every one is defined by exact token membership and its support is measured on TRAIN
text only, so the target file is a fact about the data rather than a choice made after seeing a
result.

The set is small and hand-specified ON PURPOSE. It is not tuned: these are the conversational
categories a reader would expect to see named, and the fidelity audit later checks whether each
neuron actually fires on the thing its name claims.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
SPEC = os.path.join(PROJ, "lens", "runs", "chat", "concept_spec.json")

# name -> the words that define it. Membership is by exact token, lowercased.
DEFS = {
    "is_a_question":        ["?"],
    "is_a_greeting":        ["hi", "hello", "hey", "greetings"],
    "is_a_farewell":        ["bye", "goodbye", "farewell"],
    "is_an_apology":        ["sorry", "apologize", "apologies", "apology"],
    "is_a_thanks":          ["thank", "thanks", "thankyou"],
    "mentions_the_user":    ["you", "your", "yours", "yourself"],
    "mentions_self":        ["i", "me", "my", "mine", "myself"],
    "mentions_feeling":     ["happy", "sad", "feel", "feeling", "angry", "afraid", "glad"],
    "mentions_food":        ["food", "eat", "eating", "cook", "cooking", "recipe", "meal"],
    "mentions_a_number":    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "is_a_request":         ["please", "can", "could", "would", "help"],
    "mentions_health":      ["health", "doctor", "medicine", "sick", "pain", "hurt"],
}


def build(vocab, ids, train_mask, min_support=0.005, windows=None):
    """Return (concepts, spec) with support measured the way the CONCEPT LAYER sees it.

    Support is measured PER WINDOW when windows are supplied, not per token. The first version
    measured the fraction of TOKENS containing a concept word, which dropped 8 of 12 concepts --
    "hello" is 0.023% of tokens because a greeting appears once in a 150-token conversation, but it
    is present in a much larger share of the 8-token windows the layer actually reads. Measuring
    the wrong unit made 8 real concepts look absent.
    """
    import numpy as np
    vmap = {w: i for i, w in enumerate(vocab)}
    if windows is not None and len(windows):
        unit = windows
        n = max(len(unit), 1)
    else:
        unit = ids[train_mask]
        n = max(len(unit), 1)
    concepts, spec, dropped = [], [], []
    for name, words in DEFS.items():
        tid = [vmap[w] for w in words if w in vmap]
        if not tid:
            dropped.append({"name": name, "reason": "no token in vocab"})
            continue
        supp = float(np.isin(unit, tid).any(axis=1).mean())
        if supp < min_support:
            dropped.append({"name": name, "reason": f"support {supp:.5f} < floor {min_support}"})
            continue
        concepts.append({"name": name, "ids": tid, "support": supp, "words": words})
        spec.append({"name": name, "support": supp, "n_tokens": len(tid), "words": words})
    out = {"min_support": min_support, "kept": spec, "dropped": dropped,
           "n_kept": len(spec), "n_dropped": len(dropped),
           "unit": "window" if windows is not None else "token"}
    os.makedirs(os.path.dirname(SPEC), exist_ok=True)
    json.dump(out, open(SPEC, "w"), indent=1)
    return concepts, out
