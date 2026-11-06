#!/usr/bin/env python3
"""turnend_probe.py - why does the model never stop? Measure it instead of guessing.

The stop gate said: handed the turn back 0/10, and the prediction that training the boundary token
would fix it was WRONG. Before changing anything again, measure the thing itself:

  At every REAL turn boundary in the held-out split (the exact token where the assistant stops and the
  human starts), what probability does the model assign to the boundary token?

If that probability is low, the model has not learned the boundary at all and no generation-side rule
can fix it -- the training signal has to change (weight, or a dedicated end token).
If it is high but the model never emits it while generating, then the failure is generation-side.

Also reports the top-5 tokens the model would emit there instead, and compares a ship trained with
--train-turn-end against one trained without, so the effect of that flag is visible rather than assumed.

Usage: python3 lens/scripts/turnend_probe.py <ship.npz> [<ship2.npz> ...]
"""
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
sys.path.insert(0, f"{ROOT}/lens/scripts")
os.chdir(ROOT)

import lens  # noqa: E402
import chat_data  # noqa: E402

N = int(os.environ.get("N", "400"))
SEED = int(os.environ.get("SEED", "20261018"))


def probe(ship):
    net = lens.load(ship, device="cpu")
    d = chat_data.build()
    vocab = d["vocab"]
    vmap = {w: i for i, w in enumerate(vocab)}
    ids = np.asarray(d["ids"], dtype=np.int64)
    bounds, split = d["bounds"], d["split"]
    Xva, Yva = chat_data.make_windows(ids, bounds, split, net.W, which=1)
    # the boundary token is whatever the corpus uses: <|im_end|> for the standard chat format,
    # "human" for the older corpora. Measuring the wrong one is how the first probe went wrong.
    bname = os.environ.get("BOUNDARY_TOKEN", "<|im_end|>" if "<|im_end|>" in vmap else "human")
    human_id = vmap.get(bname)
    print(f"  boundary token being measured: {bname!r} (id {human_id})")
    # THE FIX THAT MATTERS: a uniform draw over val windows lands on the boundary token only 0.5% of
    # the time (measured), so the first version of this probe was reporting P('human') at arbitrary
    # positions and calling it "not learned". Sample the windows whose TARGET IS the boundary.
    bnd = np.flatnonzero(np.asarray(Yva) == human_id)
    if len(bnd) < 20:
        print(f"  only {len(bnd)} boundary positions in val -- cannot measure")
        return {"ship": ship, "n": int(len(bnd)), "error": "too few boundaries"}
    rng = np.random.default_rng(SEED)
    idx = rng.choice(bnd, size=min(N, len(bnd)), replace=False)
    print(f"  boundary positions available in held-out data: {len(bnd):,}; sampling {len(idx)} of them")
    probs, argmax_is_human, top_other = [], 0, {}
    for i in idx:
        x = Xva[i][None, :]
        y = int(Yva[i])
        logits = np.asarray(net.forward_arrays(x)["logits"][0], dtype=np.float64)
        m = logits.max()
        p = np.exp(logits - m)
        p /= p.sum()
        probs.append(float(p[human_id]))
        if int(np.argmax(p)) == human_id:
            argmax_is_human += 1
        else:
            top_other[vocab[int(np.argmax(p))]] = top_other.get(vocab[int(np.argmax(p))], 0) + 1
    probs = np.asarray(probs)
    print(f"\nship {os.path.basename(ship)}")
    print(f"  measured at {len(probs)} REAL turn boundaries (target == the boundary token)")
    print(f"  mean P('human' | the boundary)     {probs.mean():.4f}")
    print(f"  median                             {np.median(probs):.4f}")
    print(f"  90th percentile                    {np.percentile(probs, 90):.4f}")
    print(f"  boundaries where it IS the top-1   {argmax_is_human}/{len(probs)} "
          f"({argmax_is_human/len(probs)*100:.1f}%)")
    print(f"  share above 0.05                   {(probs > 0.05).mean()*100:.1f}%")
    print(f"  what it prefers there instead: "
          f"{sorted(top_other.items(), key=lambda kv: -kv[1])[:5]}")
    verdict = ("LEARNED: the boundary is predictable at real boundaries"
               if probs.mean() > 0.2 else
               "NOT LEARNED: the boundary token is not being predicted where it truly belongs, so the "
               "training signal is the problem, not the generation rule")
    print(f"  VERDICT: {verdict}")
    return {"ship": ship, "n": int(len(probs)), "mean_p_human": float(probs.mean()),
            "top1_human_frac": argmax_is_human / len(probs)}


if __name__ == "__main__":
    ships = sys.argv[1:] or ["lens/runs/chat/te_T1_turnend_d256_24_ep24.npz",
                             "lens/runs/chat/sb_S0_d256_noattn_ep96.npz"]
    for s in ships:
        if os.path.exists(s):
            probe(s)
        else:
            print(f"(missing: {s})")