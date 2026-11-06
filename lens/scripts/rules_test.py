#!/usr/bin/env python3
"""rules_test.py - does the model LEARN THE RULE or MEMORISE THE EXAMPLES?

The user's point, and it is the right one: if a model learns rules instead of memorising, it generalises
AND needs less space, because a rule is smaller than a table of answers. Maths is the cleanest possible
test of that, so this builds a controlled arithmetic task:

  TRAIN        pairs (a, b) with a, b in 0..9, split so some pairs are held out
  HELD OUT     pairs from the SAME range the model never saw -- memorisation cannot help here
  EXTRAPOLATE  pairs with a, b in 10..19, never seen, and outside the training range entirely

Then it reports exact-answer accuracy on all three and the verdict:

  LEARNED THE RULE  high train AND high held-out AND high extrapolation
  MEMORISED         high train, low held-out (it stored the answers, not the rule)
  NEITHER           low train (it did not even fit the examples)

The gap (train - held out) is the memorisation meter. A big gap with high train accuracy is the signature
of memorising, and it is exactly what the real corpora showed earlier today (train 43.19% vs honest
4.62% at 192 epochs), so this instrument is calibrated against a known case.

Concepts: the concept layer requires named units, so this uses honest, checkable ones -- "contains a
digit", "contains the plus sign", "has two numbers", "answer is two digits" style presence tests that can
be verified from the token ids directly. Nothing here is hidden.

Usage: python3 lens/scripts/rules_test.py [--d 128] [--epochs 400] [--seed 0] [--device cpu]
"""
import argparse
import json
import os
import re
import sys
import time

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
import lens.model as M  # noqa: E402
import lens.tensor as T  # noqa: E402
from lens.concepts import Concept  # noqa: E402
import lens.opt as opt_mod  # noqa: E402

TOK = re.compile(r"\w+|[^\w\s]")


def sentence(a, b):
    return f"Human: what is {a} + {b} ? Assistant: {a + b}"


def build_vocab(lines):
    from collections import Counter
    c = Counter()
    for l in lines:
        c.update(TOK.findall(l.lower()))
    return ["<unk>"] + [w for w, _ in c.most_common()]


def encode(line, vmap, window):
    ids = [vmap.get(w, 0) for w in TOK.findall(line.lower())]
    return ids, window


def make_set(pairs, vocab, window):
    """Each example: a window ending right before the answer, and the answer token id."""
    vmap = {w: i for i, w in enumerate(vocab)}
    X, Y = [], []
    for a, b in pairs:
        full = f"Human: what is {a} + {b} ? Assistant: {a + b}"
        ids = [vmap.get(w, 0) for w in TOK.findall(full.lower())]
        if len(ids) < window + 1:
            ids = [0] * (window + 1 - len(ids)) + ids
        X.append(ids[-window - 1:-1])
        Y.append(ids[-1])
    return np.asarray(X, dtype=np.int64), np.asarray(Y, dtype=np.int64)


def concepts_for(vocab):
    """Named, checkable concepts: presence tests that can be verified from ids alone."""
    vmap = {w: i for i, w in enumerate(vocab)}
    plus = [vmap["+"]] if "+" in vmap else []
    return [Concept("has_plus_sign", plus or [0]),
            Concept("has_the_word_what", [vmap.get("what", 0)]),
            Concept("has_the_word_is", [vmap.get("is", 0)]),
            Concept("contains_question_mark", [vmap.get("?", 0)]),
            Concept("contains_two_digit_number", [vmap.get(str(i)) for i in range(10, 20)
                                                  if str(i) in vmap] or [0])]


def accuracy(net, X, Y):
    ok = 0
    for i in range(len(X)):
        logits = np.asarray(net.forward_arrays(X[i][None, :])["logits"][0], dtype=np.float64)
        ok += int(np.argmax(logits) == Y[i])
    return ok / max(len(X), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--opt", default="rms", choices=["rms", "relgrad", "sgd"],
                    help="relgrad = our scale-invariant rule from lens/opt.py: one setting "
                         "that does not depend on model size")
    a = ap.parse_args()

    # pairs: train 0..9 minus held-out 20 pairs; held out from the SAME range; extrapolation 10..19
    all_pairs = [(x, y) for x in range(10) for y in range(10)]
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(all_pairs))
    held = [all_pairs[i] for i in perm[:20]]
    train = [all_pairs[i] for i in perm[20:]]
    extra = [(x, y) for x in range(10, 20) for y in range(10, 20)]

    lines = [sentence(x, y) for x, y in all_pairs + extra]
    vocab = build_vocab(lines)
    Xtr, Ytr = make_set(train, vocab, a.window)
    Xhe, Yhe = make_set(held, vocab, a.window)
    Xex, Yex = make_set(extra, vocab, a.window)
    print(f"rules test: {len(train)} train pairs, {len(held)} held-out pairs (same range, unseen), "
          f"{len(extra)} extrapolation pairs (10..19, outside the training range)")
    print(f"  width d={a.d}, epochs {a.epochs}, device {a.device}, vocab {len(vocab)}")

    net = M.LensNet(len(vocab), concepts_for(vocab), d=a.d, window=a.window, seed=a.seed)
    if a.device == "gpu":
        import lens.backend as B
        B.use_gpu()
        net = M.LensNet(len(vocab), concepts_for(vocab), d=a.d, window=a.window, seed=a.seed)
    params = sum(p.data.size for p in net.parameters())

    opt_state = {p.name: np.zeros_like(np.asarray(p.data)) for p in net.parameters()}
    opt = opt_mod.RelativeGrad(net.parameters(), lr=a.lr, mode="relgrad")
    t0 = time.time()
    bs = min(32, len(Xtr))
    for ep in range(1, a.epochs + 1):
        order = rng.permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            idx = order[i:i + bs]
            for p in net.parameters():
                p.grad = None
            out = net.loss(Xtr[idx], Ytr[idx], use_concept_loss=True)
            loss = out[0] if isinstance(out, tuple) else out
            loss.backward()
            if a.opt == "sgd":
                for p in net.parameters():
                    p.data = p.data - a.lr * np.asarray(p.grad)
            elif a.opt == "rms":
                for p in net.parameters():
                    g = np.asarray(p.grad)
                    opt_state[p.name] = 0.9 * opt_state[p.name] + 0.1 * g * g
                    p.data = p.data - a.lr * g / (np.sqrt(opt_state[p.name]) + 1e-8)
            else:
                opt.step()
            tot += float(loss.data)
        if ep % max(1, a.epochs // 8) == 0 or ep == a.epochs:
            print(f"  ep {ep:>4}/{a.epochs}  loss {tot/max(1,len(range(0,len(Xtr),bs))):.4f}  "
                  f"({time.time()-t0:.0f}s)")

    acc_tr = accuracy(net, Xtr, Ytr)
    acc_he = accuracy(net, Xhe, Yhe)
    acc_ex = accuracy(net, Xex, Yex)
    gap = acc_tr - acc_he
    if acc_tr > 0.9 and acc_he > 0.7 and acc_ex > 0.5:
        verdict = "LEARNED THE RULE: it answers unseen pairs and pairs outside the training range"
    elif acc_tr > 0.9 and acc_he < 0.4:
        verdict = "MEMORISED: it fits the examples it saw but fails unseen pairs from the same range"
    elif acc_tr < 0.5:
        verdict = "NEITHER: it did not even fit the training examples"
    else:
        verdict = "PARTIAL: some generalisation, not enough to call it a rule"

    print(f"\n  parameters {params:,} ({params/1e6:.2f}M)")
    print(f"  exact-answer accuracy on TRAIN         {acc_tr*100:5.1f}%  ({int(acc_tr*len(Xtr))}/{len(Xtr)})")
    print(f"  exact-answer accuracy on HELD OUT      {acc_he*100:5.1f}%  ({int(acc_he*len(Xhe))}/{len(Xhe)})")
    print(f"  exact-answer accuracy on EXTRAPOLATION {acc_ex*100:5.1f}%  ({int(acc_ex*len(Xex))}/{len(Xex)})")
    print(f"  memorisation gap (train - held out)    {gap*100:5.1f} points")
    print(f"  VERDICT: {verdict}")
    json.dump({"d": a.d, "params": int(params), "epochs": a.epochs, "seed": a.seed,
               "train_acc": acc_tr, "heldout_acc": acc_he, "extrapolation_acc": acc_ex,
               "memorisation_gap": gap, "verdict": verdict,
               "n_train": len(Xtr), "n_heldout": len(Xhe), "n_extrapolation": len(Xex)},
              open(f"lens/runs/rules_d{a.d}_seed{a.seed}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())