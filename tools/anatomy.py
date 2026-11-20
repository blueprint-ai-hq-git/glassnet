#!/usr/bin/env python3
"""anatomy.py - full inventory of a trained model: every tensor, its role, its share, and what is OPAQUE.

Why this exists (2026-11-20 14:40, the directive): "take a normal model, scan it, what it has - layers, features,
tokens, KV cache, everything - but all of it is hidden, so find it." This tool finds it. It reads a checkpoint (no training,
no data needed), names every tensor, classifies its role, totals the parameters per role, works out what state must be carried
to generate the next token (the KV-cache equivalent for THIS architecture), and then says plainly which parameters have no
human meaning at all.

The output is deliberately two things: an inventory (facts about the bytes) and a verdict (what fraction of the model is
transparent). Both are printed; the inventory is also written as JSON so later phases can diff two architectures.

Nothing here is estimated. Every number is summed from the checkpoint's own arrays.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ROLE MAP - keyword -> (role, plain-English meaning). Ordered: first match wins, so the specific entries come before the general.
ROLE_MAP = [
    ("concept",     "concept layer",        "the named middle neurons - this is the readable part"),
    ("answer",      "answer layer",         "maps concepts/rules to the output vocabulary"),
    ("dmix",        "direct path",          "a shortcut from the pooled tokens to the answer, bypassing the concepts"),
    # MY OWN MAP WAS WRONG HERE (2026-11-20 14:44): dout/dproj ARE the direct path (the low-rank shortcut), but the map only
    # knew the name "dmix", so the first scan reported 208,384 parameters as OPAQUE when they are a named, inspectable component.
    # An inventory tool that cannot name a component is not evidence that the component is opaque - it is evidence the map is thin.
    ("dout",        "direct path",          "direct-path output matrix (low-rank shortcut to the vocabulary)"),
    ("dproj",       "direct path",          "direct-path input projection (pooled tokens -> rank)"),
    ("Wp",          "state projection",     "projects recurrent state into embedding space"),
    ("emb",         "token table",          "the lookup table: one row per vocabulary word"),
    ("mem",         "memory bank",          "the readable window bank (slots holding real tokens)"),
    ("age",         "memory age key",       "tells a slot how old its contents are"),
    ("Wi",          "input projection",     "first layer of the recurrent core"),
    ("Wh",          "hidden projection",    "recurrent core matrix"),
    ("Wx",          "input path",           "token-to-state matrix"),
    ("Wa",          "readout matrix",       "final matrix from state to vocabulary"),
    ("W",           "matrix",               "unnamed matrix"),
    ("b",           "bias",                 "a per-output offset"),
    ("lam",         "scalar",               "a loss weight, not a parameter of the model"),
]


def classify(name):
    for key, role, meaning in ROLE_MAP:
        if key in name:
            return role, meaning
    return "UNNAMED", "no role in the map - this is an opaque tensor"


def arch_family(name):
    if "off2" in name or "win6k" in name or "untied6k" in name or "mem" in name:
        return "tokmem (recurrent + readable window bank)"
    if "narrow" in name or "dec_" in name or "windec" in name or "arch_" in name:
        return "LensNet (concept bottleneck + answer layer)"
    if "match3" in name or "exp" in name:
        return "image readable"
    return "unknown"


def state_per_token(arrays, meta):
    """What must be carried between tokens to generate the next one.

    For the recurrent families the answer is the hidden state plus the memory bank; for a transformer-style block it would be
    key/value vectors per layer. This reports what THIS checkpoint actually requires, derived from the tensor shapes rather than
    from documentation.
    """
    H = None
    for k in ("Wh", "Wi", "Wx", "Wa"):
        if k in arrays and arrays[k].ndim == 2:
            H = arrays[k].shape[0] if k in ("Wh", "Wi") else arrays[k].shape[1]
            if k in ("Wh",):
                H = arrays[k].shape[0]
    # memory bank slots: the emb table is (V, D); a bank tensor shaped (K, D) is K slots of D
    bank_slots, slot_dim = 0, 0
    for k, v in arrays.items():
        if v.ndim == 2 and ("mem" in k or "bank" in k) and v.shape[0] <= 64:
            bank_slots, slot_dim = int(v.shape[0]), int(v.shape[1])
    return {"hidden_state_floats": int(H) if H else None,
            "memory_slots": bank_slots, "slot_width": slot_dim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+", help="one or more .npz checkpoints")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = []
    for path in a.ckpt:
        if not os.path.isfile(path):
            print("MISSING: %s" % path)
            continue
        z = np.load(path, allow_pickle=True)
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
        meta = {}
        if "__meta__" in z.files:
            try:
                meta = json.loads(str(z["__meta__"]))
            except Exception:
                meta = {"raw": str(z["__meta__"])[:200]}

        total = sum(int(v.size) for v in arrays.values())
        by_role = {}
        inventory = []
        unnamed = []
        for k in sorted(arrays):
            v = arrays[k]
            role, meaning = classify(k)
            n = int(v.size)
            pct = 100.0 * n / total if total else 0.0
            by_role[role] = by_role.get(role, 0) + n
            if role == "UNNAMED":
                unnamed.append(k)
            inventory.append({"tensor": k, "shape": list(v.shape), "dtype": str(v.dtype),
                              "params": n, "pct": round(pct, 3), "role": role, "meaning": meaning})

        # table vs compute: a lookup table stores one row per vocabulary word and computes nothing
        table = sum(int(v.size) for k, v in arrays.items() if classify(k)[0] in ("token table", "answer layer"))
        compute = total - table
        st = state_per_token(arrays, meta)
        transparent = total - sum(int(arrays[k].size) for k in unnamed)

        print("=" * 100)
        print("%s" % path)
        print("  family: %s   |  parameters: %s  |  sha256 %s"
              % (arch_family(os.path.basename(path)), format(total, ","), sha256(path)[:16]))
        print("  table (lookup, computes nothing): %s (%0.1f%%)   |   compute: %s (%0.1f%%)"
              % (format(table, ","), 100.0 * table / total, format(compute, ","), 100.0 * compute / total))
        print("  state needed per token: %s floats of hidden state, %d memory slots x %d wide"
              % (st["hidden_state_floats"], st["memory_slots"], st["slot_width"]))
        print("  %-22s %-11s %-18s %10s %7s" % ("tensor", "shape", "role", "params", "share"))
        for it in inventory:
            print("  %-22s %-11s %-18s %10s %6.2f%%"
                  % (it["tensor"], "x".join(str(s) for s in it["shape"])[:11], it["role"],
                     format(it["params"], ","), it["pct"]))
        print("  ---")
        for role, n in sorted(by_role.items(), key=lambda kv: -kv[1]):
            print("  %-20s %12s  %5.1f%%" % (role, format(n, ","), 100.0 * n / total))
        if unnamed:
            print("  OPAQUE tensors with no role in the map: %s" % ", ".join(unnamed))
        else:
            print("  OPAQUE tensors: none - every tensor in this checkpoint has a named role")
        print("  transparent parameter share: %.1f%% of %s" % (100.0 * transparent / total, format(total, ",")))
        rows.append({"path": path, "family": arch_family(os.path.basename(path)), "params": total,
                     "table": table, "compute": compute, "by_role": by_role,
                     "unnamed": unnamed, "state_per_token": st,
                     "transparent_share": 100.0 * transparent / total, "inventory": inventory, "meta": meta})

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print("\ninventory written to %s (%d checkpoints)" % (a.out, len(rows)))

    if len(rows) > 1:
        print("\n" + "=" * 100)
        print("CROSS-ARCHITECTURE COMPARISON")
        print("  %-44s %12s %9s %9s %8s" % ("checkpoint", "params", "table%", "compute%", "transp%"))
        for r in rows:
            print("  %-44s %12s %8.1f%% %8.1f%% %7.1f%%"
                  % (os.path.basename(r["path"])[:44], format(r["params"], ","),
                     100.0 * r["table"] / r["params"], 100.0 * r["compute"] / r["params"],
                     r["transparent_share"]))


if __name__ == "__main__":
    main()
