"""chat_concepts_char.py — concept targets for CHARACTER-level windows.

The concept definitions are unchanged (the same word lists). What changes is how a target is
computed: at character level the layer reads a window of characters, so a concept fires when the
window's TEXT contains one of the concept's words.

APPROXIMATION, STATED PLAINLY
-----------------------------
Boundary checking is simple: a word matches when it appears as a substring bounded by characters
that are not letters or apostrophes. That means "you" matches in "you" and "you're" but not in
"your" -- except that "your" IS in the concept's own word list, so it is covered by its own entry.
The approximation is that this is substring matching on lowercased text, not tokenisation. It is
close enough to be a fair target and it is written down rather than assumed.
"""
import json
import os
import re

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
SPEC = os.path.join(PROJ, "lens", "runs", "chat", "concept_spec_char.json")


def build(vocab, ids, train_mask, min_support=0.005, windows=None, defs=None):
    """windows: (N, W) character ids. Returns (concepts, spec) with per-window support."""
    import numpy as np
    from chat_concepts_wide import DEFS as WIDE_DEFS
    defs = defs or WIDE_DEFS
    unit = windows if (windows is not None and len(windows)) else None
    if unit is None:
        raise ValueError("character concepts need windows")

    # decode each window once, then match by regex with boundaries
    texts = ["".join(vocab[i] if 0 <= i < len(vocab) else " " for i in w) for w in unit]
    concepts, spec, dropped = [], [], []
    for name, words in defs.items():
        pat = re.compile("|".join(r"(?<![a-z'])" + re.escape(w) + r"(?![a-z'])" for w in words))
        hits = np.fromiter((1.0 if pat.search(t) else 0.0 for t in texts),
                           dtype=np.float32, count=len(texts))
        supp = float(hits.mean())
        if supp < min_support:
            dropped.append({"name": name, "reason": f"support {supp:.5f} < floor {min_support}"})
            continue
        concepts.append({"name": name, "words": words, "support": supp, "pattern": pat.pattern})
        spec.append({"name": name, "support": supp, "n_words": len(words), "words": words})
    out = {"min_support": min_support, "kept": spec, "dropped": dropped,
           "n_kept": len(spec), "n_dropped": len(dropped), "n_defined": len(defs),
           "unit": "character-window", "matching": "substring with letter/apostrophe boundaries"}
    os.makedirs(os.path.dirname(SPEC), exist_ok=True)
    json.dump(out, open(SPEC, "w"), indent=1)
    return concepts, out


def targets_for(texts, concepts):
    """(N, K) float targets for a batch of decoded window texts."""
    import numpy as np
    out = np.zeros((len(texts), len(concepts)), dtype=np.float32)
    for k, c in enumerate(concepts):
        pat = re.compile(c["pattern"])
        for i, t in enumerate(texts):
            if pat.search(t):
                out[i, k] = 1.0
    return out


def decode_windows(vocab, windows):
    return ["".join(vocab[i] if 0 <= i < len(vocab) else " " for i in w) for w in windows]
