"""code_concepts.py — named concepts for CODE. Structural, measurable, and readable.

A concept layer for code should name the things a programmer would name when describing what a
function does. Every concept here is defined by exact token membership in the code vocabulary and
its support is measured per WINDOW on TRAIN text only, so the target file is a fact about the data.

These are deliberately ordinary: `has_loop`, `has_return`, `has_condition`. A reader can check any
one of them against the code by eye, which is the whole point of the layer.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
SPEC = os.path.join(PROJ, "lens", "runs", "code", "concept_spec.json")

DEFS = {
    # ORTHOGONALITY RULE (2026-10-10): every anchor token belongs to EXACTLY ONE concept.
    # Shared anchors made formulas collinear (has_a_tuple & has_a_call both owned "(" -> |corr|
    # 0.99) so no neuron could be label-honest. Each family below is disjoint by construction;
    # the pre-gate (audit_formula_orthogonality.py) measures what the data does to them anyway.

    # --- control flow (exclusive anchors)
    "has_a_loop":            ["for", "while"],
    "has_a_condition":       ["if", "elif", "else"],
    "has_break_or_continue": ["break", "continue"],
    "has_return":            ["return"],                    # "return" is return's alone
    "has_error_handling":    ["try", "except", "finally", "raise"],

    # --- definitions
    "defines_a_function":    ["def"],                       # "def" is def's alone
    "defines_a_class":       ["class"],
    "has_import":            ["import", "from"],
    "is_recursive_shape":    ["recursive", "recursion", "self_call"],  # no longer twins with def/return

    # --- data structures (brackets now belong to ONE owner each)
    "has_a_list":            ["list"],                      # "[" and "]" belong to has_indexing
    "has_a_dict":            ["dict", "{", "}"],    "has_a_set":             ["set"],    "has_a_number":          ["int", "float", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "has_a_boolean":         ["true", "false", "bool"],
    "has_none":              ["none", "null"],              # "none" is none's alone

    # --- operations
    "has_addition":          ["+"],
    "has_subtraction":       ["-"],
    "has_multiplication":    ["*"],
    "has_division":          ["/"],
    "has_comparison":        ["==", "!=", "<=", ">=", "<", ">"],    "has_indexing":          ["[", "]"],
    # --- input / output    "writes_output":         ["print", "output", "write", "echo"],   # "return" removed
    "has_a_comment":         ["#"],
    "appends_to_something":  ["append", "extend", "add"],
    "sorts_something":       ["sort", "sorted"],
    "uses_len":              ["len"],    "uses_enumerate":        ["enumerate"],
    "uses_zip":              ["zip"],
    "uses_map_or_filter":    ["map", "filter"],

    # --- common problem shapes (word twins removed)    "mentions_a_string":     ["string", "word", "characters"],
    "mentions_a_number":     ["number", "value", "sum", "total"],  # "integer" removed    "mentions_empty":        ["empty", "zero", "nothing", "void"],  # "none" removed
    "mentions_max_or_min":   ["max", "min", "maximum", "minimum"],    "mentions_order":        ["order", "ascending", "descending"],  # "sorted" removed
}


def build(vocab, ids, train_mask, min_support=0.005, windows=None):
    """Return (concepts, spec). Support measured PER WINDOW when windows are supplied."""
    import numpy as np
    vmap = {w: i for i, w in enumerate(vocab)}
    unit = windows if (windows is not None and len(windows)) else ids[train_mask]
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
           "n_kept": len(spec), "n_dropped": len(dropped), "n_defined": len(DEFS),
           "unit": "window" if windows is not None else "token"}
    os.makedirs(os.path.dirname(SPEC), exist_ok=True)
    json.dump(out, open(SPEC, "w"), indent=1)
    return concepts, out
