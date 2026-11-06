#!/usr/bin/env python3
"""CONCEPT LEAKAGE CHECK — does a concept secretly encode something other than its own label?

WHY (adopted from research/eval-standards-and-provenance.md, item 6)
Concept leakage (Mahinpei et al. 2021, arXiv 2106.13314) is the field's recurring failure mode: a
concept's learned representation carries information that was never meant to be in it. The published
work MEASURES it and reports it; nothing in the verified literature enforces it as an admission gate.
Our model's concepts are hand-defined formulas over tokens, so the risk is concrete and testable: a
concept neuron can end up encoding "which data source this came from" or "how long this conversation
is" rather than the token pattern its name claims.

WHAT IS MEASURED, per concept
  own-label AUROC   -- can this concept's activation tell its OWN formula's presence from absence?
  confound AUROCs   -- can the same activation tell apart things it was NOT asked about:
                       (a) corpus position (the corpora are concatenated BY SOURCE, so this is a
                           source proxy),
                       (b) conversation length bucket,
                       (c) position inside the conversation (early vs late).
                       Each is reported raw AND length-residualised, because a concept that merely
                       tracks length would otherwise look like a leak in the other two columns.
  control           -- the same AUROCs computed on a RANDOM PERMUTATION of the activations. The control
                       must read ~0.50; if it does not, the meter is broken and nothing it says counts.

BARS (pre-registered 2026-10-16, before running on any ship)
  LEAKY  if a residualised confound AUROC >= 0.65 AND within 0.10 of the concept's own-label AUROC.
  CLEAN  otherwise. Both numbers are always printed next to the verdict.

Usage: bash lens/scripts/run_concept_leakage.sh <ship.npz> [corpus-dir] [n-windows]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet")
import chat_data  # noqa: E402
import lens  # noqa: E402

WIN = 32
LEAK_BAR = 0.65
CLOSE_TO_OWN = 0.10


def auroc(scores, labels):
    """Rank-based AUROC (no scipy in this project: numpy only, per the from-scratch rules)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def residualise(x, design):
    """Remove the column space of `design` from x by least squares (numpy only)."""
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ beta


def main():
    ship = sys.argv[1]
    corpus = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("CHAT_DATA_DIR")
    n_take = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    if corpus:
        os.environ["CHAT_DATA_DIR"] = corpus
    d = chat_data.build()
    ids, bounds, split, vocab = d["ids"], d["bounds"], d["split"], d["vocab"]
    bounds, split = np.asarray(bounds), np.asarray(split)

    # rebuild the window list WITH the provenance of each window (conv index, start, conv length)
    X, conv_i, conv_len, start_in_conv = [], [], [], []
    for j in range(len(bounds) - 1):
        if split[j] != 0:
            continue
        seg = ids[bounds[j]:bounds[j + 1]]
        for s in range(0, len(seg) - WIN - 1, WIN // 2):
            X.append(seg[s:s + WIN])
            conv_i.append(j)
            conv_len.append(len(seg))
            start_in_conv.append(s)
    X = np.asarray(X, dtype=np.int64)
    conv_i = np.asarray(conv_i)
    conv_len = np.asarray(conv_len)
    start_in_conv = np.asarray(start_in_conv)
    if len(X) > n_take:
        sel = np.random.default_rng(0).choice(len(X), n_take, replace=False)
        sel.sort()
        X, conv_i, conv_len, start_in_conv = X[sel], conv_i[sel], conv_len[sel], start_in_conv[sel]
    print(f"  corpus {os.environ.get('CHAT_DATA_DIR')}   windows sampled {len(X):,}")

    net = lens.load(ship)
    h = np.asarray(net.forward_arrays(X)["concepts"], dtype=np.float64)
    K = h.shape[1]
    names = [c.name for c in net.concepts]
    cids = [int(c.ids[0]) for c in net.concepts]

    # confound labels
    pos = conv_i.astype(np.float64) / max(len(bounds) - 1, 1)
    conf_pos = pos > np.median(pos)                      # corpus position (source proxy)
    conf_len = conv_len > np.median(conv_len)            # long conversation?
    early = start_in_conv < (conv_len - WIN) / 2.0
    conf_early = early                                   # early vs late in the conversation

    design = np.column_stack([np.ones(len(X)), np.log1p(conv_len.astype(np.float64))])
    perm = np.random.default_rng(1).permutation(len(X))

    print(f"  ship {os.path.basename(ship)}   concepts {K}")
    print(f"  {'concept':<26} {'own-label':>9} {'pos':>7} {'len':>7} {'early':>7} | "
          f"{'res.pos':>8} {'res.len':>8} {'res.early':>9} {'control':>8}  verdict")
    rows = []
    for c in range(K):
        act = h[:, c]
        resid = residualise(act, design)
        own = auroc(act, (X == cids[c]).any(axis=1))
        raw = {"pos": auroc(act, conf_pos), "len": auroc(act, conf_len), "early": auroc(act, conf_early)}
        res = {"pos": auroc(resid, conf_pos), "len": auroc(resid, conf_len),
               "early": auroc(resid, conf_early)}
        ctrl = auroc(act[perm], conf_pos)
        worst = max(res.values())
        leaky = worst >= LEAK_BAR and worst >= (own - CLOSE_TO_OWN)
        rows.append({"concept": names[c], "own_label_auroc": own, "raw": raw, "residual": res,
                     "control": ctrl, "worst_residual": worst, "verdict": "LEAKY" if leaky else "clean"})
        flag = "  <-- LEAKY" if leaky else ""
        print(f"  {names[c][:26]:<26} {own:>9.3f} {raw['pos']:>7.3f} {raw['len']:>7.3f} "
              f"{raw['early']:>7.3f} | {res['pos']:>8.3f} {res['len']:>8.3f} {res['early']:>9.3f} "
              f"{ctrl:>8.3f}  {rows[-1]['verdict']}{flag}")
    ctrls = [r["control"] for r in rows]
    leaky = [r for r in rows if r["verdict"] == "LEAKY"]
    print()
    print(f"  control check: permuted-activation AUROC min {min(ctrls):.3f} max {max(ctrls):.3f} "
          f"(must sit near 0.50; a control far from 0.50 means the meter is broken)")
    print(f"  VERDICT (computed from the pre-registered bar, worst residualised confound AUROC "
          f">= {LEAK_BAR:.2f}): {len(leaky)} of {K} concepts LEAKY")
    for r in leaky:
        print(f"    - {r['concept']}: own {r['own_label_auroc']:.3f}  worst confound "
              f"{r['worst_residual']:.3f}")
    dest = ("/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/leakage_"
            f"{os.path.basename(ship).replace('.npz', '')}.json")
    json.dump({"ship": ship, "n_windows": int(len(X)), "bars": {"leak": LEAK_BAR,
                                                              "close_to_own": CLOSE_TO_OWN},
               "rows": rows}, open(dest, "w"), indent=1)
    print(f"  written: {dest}")


if __name__ == "__main__":
    main()