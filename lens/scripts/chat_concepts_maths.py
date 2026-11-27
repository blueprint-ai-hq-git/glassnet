"""MATHS CONCEPTS - the labels a maths model should be reading, instead of chat topics.

WHY THIS FILE EXISTS
--------------------
tools/forensics_math_model.py looked inside the maths solver and found its 29 named concepts were all CHAT concepts
(mentions_self, is_a_question, mentions_a_colour, mentions_clothing) applied to arithmetic text, and that the direct route
supplied 78.7% of the logits while the named route alone agreed with the model's answer only 8.9% of the time. The readable part of
the model was describing the wrong subject and was not deciding the answer.

These are word-set features over the input window, the same mechanism the chat concepts use, so nothing about the architecture
changes - only what the model is asked to name. A label is only kept if its words actually exist in this vocabulary; the builder
drops and reports the rest rather than pretending.
"""
import os as _os_mod

PROJ = _os_mod.path.dirname(_os_mod.path.dirname(_os_mod.path.dirname(_os_mod.path.abspath(__file__))))
SPEC = PROJ + "/lens/runs/chat/concept_spec_maths.json"

_d2 = [str(n) for n in range(10, 100)]
_d3 = [str(n) for n in range(100, 1000)]
_d4 = [str(n) for n in range(1000, 10000)]

DEFS = {
    # the operation being performed
    "has_plus": ["+", "plus", "added", "sum"],
    "has_minus": ["-", "minus", "subtracted", "difference"],
    "has_times": ["*", "x", "times", "multiply", "multiplied", "product"],
    "has_divide": ["/", "divide", "divided", "quotient", "per"],
    "has_equals": ["=", "equals", "is"],
    # the shape of the working
    "has_fraction": ["frac", "\\frac", "numerator", "denominator", "half", "third", "quarter"],
    "has_power": ["^", "squared", "cubed", "exponent", "power"],
    "has_root": ["sqrt", "\\sqrt", "root"],
    "has_parentheses": ["(", ")"],
    "has_money": ["$", "dollars", "cents", "price", "cost", "paid", "earn", "spent"],
    "has_unit_measure": ["miles", "hours", "minutes", "seconds", "kg", "pounds", "meters", "feet", "liters", "gallon"],
    # concluding language, which marks where the final answer is about to appear
    "says_answer": ["answer", "ans"],
    "says_therefore": ["therefore", "thus", "hence", "finally", "so"],
    # number size inside the window
    "has_two_digit": _d2,
    "has_three_digit": _d3,
    "has_four_digit": _d4,
    "has_many_digits": ["digits", "digit", "number", "numbers", "integer"],
    # problem framing
    "is_a_question": ["?", "how", "what", "many", "much"],
    "mentions_a_person": ["he", "she", "his", "her", "they", "them", "name"],
    "is_comparison": ["more", "less", "greater", "fewer", "than", "least", "most"],
    "is_ratio": ["ratio", "percent", "%%", "proportion", "rate"],
}

import json as _json
import os as _os

import numpy as _np


def build(vocab, ids, train_mask, min_support=0.005, windows=None):
    vmap = {w: i for i, w in enumerate(vocab)}
    unit = windows if (windows is not None and len(windows)) else ids[train_mask]
    concepts, spec, dropped = [], [], []
    for name, words in DEFS.items():
        tid = [vmap[w] for w in words if w in vmap]
        if not tid:
            dropped.append({"name": name, "reason": "no token in vocab"})
            continue
        supp = float(_np.isin(unit, tid).any(axis=1).mean())
        if supp < min_support:
            dropped.append({"name": name, "reason": "support %.5f < floor %s" % (supp, min_support)})
            continue
        concepts.append({"name": name, "ids": tid, "support": supp, "words": words})
        spec.append({"name": name, "support": supp, "n_tokens": len(tid), "words": words})
    out = {"min_support": min_support, "kept": spec, "dropped": dropped,
           "n_kept": len(spec), "n_dropped": len(dropped),
           "unit": "window" if windows is not None else "token",
           "n_defined": len(DEFS)}
    _os.makedirs(_os.path.dirname(SPEC), exist_ok=True)
    _json.dump(out, open(SPEC, "w"), indent=1)
    return concepts, out
