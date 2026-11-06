"""build_fidelity.py — measure, for every concept in a checkpoint, how well its neuron actually
tracks its own name. Writes a table that explain() attaches to every explanation.

WHY THIS EXISTS (E18, measured 2026-10-09)
  explain() listed the top concepts by contribution and presented each NAME as a cause. The meaning
  audit showed that is not safe: on the lam_concept=0.00 medical checkpoint the neurons' mean
  correlation with their own named concept was 0.033 with ZERO concepts above 0.70 — the names were
  decorative. Printing "I said X because of rule is_bright" when is_bright correlates 0.05 with the
  neuron is a false explanation, not a readable one.

  So each concept gets a FIDELITY score: its correlation with its own formula, plus the best OTHER
  concept's correlation (its nearest rival). A name is only presented as a cause when it earns it.

THE TABLE (per checkpoint, per concept)
  own_corr     correlation of the neuron with its OWN concept formula
  best_other   the strongest correlation with any OTHER concept formula
  margin       own_corr - best_other   (positive = the name is the best available description)
  rank         where its own concept places among all concepts by |corr|
  verdict      strong   own_corr >= 0.70 and margin > 0.10
               honest   own_corr >= 0.40 and rank <= 3
               weak     own_corr >= 0.20
               decorative  own_corr < 0.20  -- must NOT be shown as a cause

GATES (pre-registered)
  d1  every concept gets a fidelity row (no concept silently skipped)
  d2  the table is written next to the checkpoint so explain() can find it deterministically
  d3  a checkpoint whose concepts are mostly decorative must be REPORTED as such in the table
"""
import argparse
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

from lens import backend as B                             # noqa: E402

MAXN = int(os.environ.get("FID_N", "3000"))


def corr(a, b):
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def verdict_of(own, margin, rank):
    if own >= 0.70 and margin > 0.10:
        return "strong"
    if own >= 0.40 and rank <= 3:
        return "honest"
    if own >= 0.20:
        return "weak"
    return "decorative"


def build(net, X, Q=None, tag="", cap_ids=None):
    """Fidelity table for a loaded vision / VL / see_talk net.

    X = images. Q = question ids (VL) or None. cap_ids = caption prefix ids, required by models
    whose forward needs a text condition (SeeTalkNet) — it conditions the concept layer on the
    caption words, so the activations being audited are the ones that model actually produces.
    """
    names = [c.name for c in net.concepts]
    n = len(names)
    if cap_ids is not None:
        T = np.stack([np.asarray(c.labels(X), dtype=np.float32) for c in net.concepts], axis=1)
        H = B.to_cpu(net.forward_arrays(X, cap_ids)["concepts"])
    elif Q is None:
        T = np.stack([np.asarray(c.labels(X), dtype=np.float32) for c in net.concepts], axis=1)
        H = B.to_cpu(net.forward_arrays(X)["concepts"])
    else:
        T = net.concept_targets(X, Q)
        H = B.to_cpu(net.forward_arrays(X, Q)["concepts"])
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = corr(H[:, i], T[:, j])
    rows = []
    for i in range(n):
        own = float(M[i, i])
        order = np.argsort(-np.abs(M[i]))
        rank = int(np.where(order == i)[0][0]) + 1
        jb = int(order[0]) if order[0] != i else (int(order[1]) if n > 1 else i)
        best_other = float(M[i, jb])
        margin = own - abs(best_other)
        rows.append({"name": names[i], "own_corr": round(own, 4),
                     "best_other": round(best_other, 4), "best_other_name": names[jb],
                     "margin": round(margin, 4), "rank": rank,
                     "verdict": verdict_of(own, margin, rank)})
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {"tag": tag, "n_concepts": n, "n_samples": int(len(X)),
            "counts": counts, "concepts": rows,
            "note": ("fidelity = how well each neuron tracks its own named concept. A concept whose "
                     "verdict is 'decorative' must not be presented by explain() as a cause.")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["vl", "vision", "seetalk"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="pathmnist")
    ap.add_argument("--color", action="store_true")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import image_concepts
    import image_concepts_adaptive as ic_mod
    from lens.vision import load_vision

    if a.kind in ("vl", "vision"):
        import vl_data
        import vl_data3
        base = vl_data.prepare_vqa(a.data, max_train=None, seed=0, color=a.color)
        d = vl_data3.prepare(base, "medical", seed=0)
        Xtr = np.asarray(d["Xtr"], dtype=np.float32)
        kept, _ = ic_mod.build(Xtr, min_support=0.02, verbose=False)
        image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in kept]
        if a.color:
            import colour_concepts
            ck, _ = colour_concepts.build(Xtr, min_support=0.02, verbose=False)
            image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in ck]
        net, _meta = load_vision(a.ckpt, device="cpu")
        X = np.asarray(d["Xte"][:MAXN], dtype=np.float32)
        Q = np.asarray(d["Qte"][:MAXN], dtype=np.int64) if a.kind == "vl" else None
        tbl = build(net, X, Q, tag=os.path.basename(a.ckpt))
    else:
        import train_seetalk
        net, meta = train_seetalk.SeeTalkNet.load(a.ckpt)
        # see_talk consumes IMAGES; its concept labels come from the pixels themselves, so the
        # images are all we need. This branch previously called vl_data3.prepare (the medical VQA
        # wrapper, which was not even imported here) — an unnecessary and broken detour.
        import vl_data
        base = vl_data.prepare_vqa(a.data, max_train=None, seed=0, color=a.color)
        X = np.asarray(base["Xte"][:MAXN], dtype=np.float32)
        # SeeTalkNet conditions the concept layer on a caption prefix; build the same "this is a"
        # prefix the product uses so the audited activations are the product's own.
        vi = {w: i for i, w in enumerate(meta["vocab"])}
        pref = [vi.get("<pad>", 0), vi.get("<pad>", 0), vi.get("<bos>", 0),
                vi.get("this", 0), vi.get("is", 0), vi.get("a", 0)]
        cap = np.asarray([pref] * len(X), dtype=np.int64)
        tbl = build(net, X, None, tag=os.path.basename(a.ckpt), cap_ids=cap)

    out = a.out or (os.path.splitext(a.ckpt)[0] + "_fidelity.json")
    json.dump(tbl, open(out, "w"), indent=1)
    print(f"  {tbl['tag']}: {tbl['n_concepts']} concepts, {tbl['n_samples']} samples")
    for k, v in sorted(tbl["counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<12} {v}")
    print(f"  wrote {out}")
    print(f"  d1 every concept has a row: "
          f"{'PASS' if len(tbl['concepts']) == tbl['n_concepts'] else 'FAIL'}")
    d2 = os.path.exists(out)
    print(f"  d2 written next to the checkpoint: {'PASS' if d2 else 'FAIL'} ({out})")
    dec = tbl["counts"].get("decorative", 0)
    print(f"  d3 {dec} decorative concept(s) — explain() must not present those as causes")
