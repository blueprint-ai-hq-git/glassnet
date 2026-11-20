#!/usr/bin/env python3
"""memg.py - MEMORISE VERSUS GENERALISE, with a trap dataset that separates them by construction.

the ask: "it automatically differentiates between generalisation and memorisation - it can memorize when it is needed and
generalise when it is needed". This file builds the two halves and measures them apart, because a dataset where both work tells you
nothing.

THE TRAP DATASET (synthetic, generated here, no external data):
  * RULE examples: two-digit addition. The answer is computed by a rule that exists for EVERY pair, so held-out pairs are answerable -
    a model that generalises gets them right, a model that memorises cannot (it has never seen them).
  * FACT examples: a 3-digit key maps to a 2-digit value, and EVERY KEY APPEARS EXACTLY ONCE. That mapping exists for one key only,
    so it is answerable ONLY by storing it. No amount of generalisation recovers it.

So: rules test generalisation on unseen inputs; facts test memorisation of arbitrary unique inputs. Same model, same training run.

THE SWITCH (the part that decides):
  store a context -> the model's own prediction at that context is WRONG and the context appears EXACTLY ONCE in training.
  Rationale: a context seen once cannot be generalised from and the model got it wrong, so the only way to be right is to remember it.
  A context that appears many times is evidence of a rule, so it is left to the weights.
  At inference: exact-context hit -> use the stored answer; miss -> use the model.

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 memorisation: fact accuracy with the switch is at least 3x fact accuracy without it (the model alone cannot know unique keys).
  P2 generalisation is NOT hurt: held-out rule accuracy changes by no more than 1.0 point with the switch on.
  P3 no collision: at most 1% of held-out rule contexts match any stored fact context, so P2 cannot be an artefact of the store
     stealing rule questions.
FALSIFIER: if held-out rule accuracy falls by more than 2.0 points with the store enabled, the switch is a TRADE (memorisation bought
  at the cost of generalisation) and must be reported as one, not as a free win.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT", ".")
sys.path.insert(0, ROOT)

from lens.glass_lm import GlassLM, Momentum

# tiny synthetic vocabulary, defined here so the data is fully self-describing
DIG = {str(d): d for d in range(10)}          # ids 0-9 are the digits 0-9
PLUS = 10
EQ = 11
KEY = 12
VAL = 13
PAD = 14
V = 15


def d(x):
    return x % 10


def make_rule(a, b):
    """[a1 a2 PLUS b1 b2 EQ c1 c2 c3] - two-digit addition, result is the target part."""
    c = a + b
    return [d(a // 10), d(a % 10), PLUS, d(b // 10), d(b % 10), EQ, d(c // 100), d((c // 10) % 10), d(c % 10)]


def make_fact(k, v):
    """[KEY k1 k2 k3 VAL v1 v2] - an arbitrary unique mapping."""
    return [KEY, d(k // 100), d((k // 10) % 10), d(k % 10), VAL, d(v // 10), d(v % 10)]


def build(rng, n_rule_train=600, n_rule_test=200, n_fact=200):
    seen = set()
    rule_train, rule_test = [], []
    while len(rule_train) < n_rule_train:
        a, b = int(rng.integers(0, 100)), int(rng.integers(0, 100))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rule_train.append(make_rule(a, b))
    while len(rule_test) < n_rule_test:
        a, b = int(rng.integers(0, 100)), int(rng.integers(0, 100))
        if (a, b) in seen:            # held-out pairs only: a memoriser cannot answer these
            continue
        seen.add((a, b))
        rule_test.append(make_rule(a, b))
    keys = rng.choice(np.arange(100, 1000), size=n_fact, replace=False)   # every key appears exactly once
    facts = [make_fact(int(k), int(rng.integers(0, 100))) for k in keys]
    return rule_train, rule_test, facts


def rule_answer_start(seq):
    """Index of the first answer token in a rule sequence (after EQ)."""
    return seq.index(EQ) + 1


def fact_answer_start(seq):
    return seq.index(VAL) + 1


def to_batch(seqs, T):
    """Left-pad/right-pad sequences to T and return (inputs, targets). Target at position t is the token at t+1."""
    X = np.full((len(seqs), T + 1), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s if len(s) <= T + 1 else s[:T + 1]
        X[i, :len(s)] = s
    return X[:, :T], X[:, 1:]


def score(lm, seqs, answer_start, store=None, T=10):
    """Accuracy on the ANSWER tokens only: teacher-forced, argmax per position, store consulted on exact context matches."""
    right = 0
    total = 0
    hits = 0
    for s in seqs:
        start = answer_start(s)
        for t in range(start, len(s)):
            ctx = tuple(s[:t])
            total += 1
            pred = None
            if store is not None:
                pred = store.get(ctx)
                if pred is not None:
                    hits += 1
            if pred is None:
                toks, _ = to_batch([s[:t]], T)
                logits, _ = lm.forward(toks[:, :max(1, t)])
                pred = int(np.argmax(logits[0, max(0, t - 1), :]))
            if pred == s[t]:
                right += 1
    return right / max(total, 1), total, hits


def build_store(lm, facts, T=10):
    """The switch: store a context only if it appears ONCE in training AND the model gets it wrong."""
    counts = {}
    for s in facts:
        for t in range(1, len(s)):
            counts[tuple(s[:t])] = counts.get(tuple(s[:t]), 0) + 1
    store = {}
    for s in facts:
        for t in range(1, len(s)):
            ctx = tuple(s[:t])
            if counts[ctx] != 1:
                continue
            toks, _ = to_batch([s[:t]], T)
            logits, _ = lm.forward(toks[:, :max(1, t)])
            pred = int(np.argmax(logits[0, max(0, t - 1), :]))
            if pred != s[t]:                      # wrong and unique -> nothing to generalise from, so remember it
                store[ctx] = s[t]
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rule-train", type=int, default=600,
                    help="how many rule pairs the model sees. 2026-11-20 21:10: the generalisation run failed at 600 pairs with a "
                         "17-point train/held-out gap, so the single pre-registered change is to raise this and change NOTHING else.")
    ap.add_argument("--n-fact", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    rule_tr, rule_te, facts = build(rng, n_rule_train=a.n_rule_train, n_fact=a.n_fact)
    mixed = rule_tr + facts
    print("data: %d rule-train, %d rule-test (HELD-OUT pairs), %d single-occurrence facts, vocab %d"
          % (len(rule_tr), len(rule_te), len(facts), V))
    print("  rules need generalisation (unseen pairs); facts CANNOT be generalised - each key appears exactly once")

    lm = GlassLM(V, d=a.d, layers=a.layers, n_heads=a.heads, d_mlp=2 * a.d, T=a.T, seed=a.seed, emb_rank=4)
    print("model: %s parameters" % format(lm.n_params(), ","))
    opt = Momentum(lm.p, lr=a.lr, momentum=0.9)
    rr = np.random.default_rng(a.seed + 1)
    for step in range(1, a.steps + 1):
        idx = rr.integers(0, len(mixed), size=a.batch)
        # BUG FIXED (2026-11-20 20:20): rule rows are 9 tokens and fact rows are 7, so np.stack raised "all input arrays must
        # have the same shape". to_batch already pads variable-length sequences, so the stack was pointless as well as wrong.
        chunk = [mixed[i] for i in idx]
        toks, tgts = to_batch(chunk, a.T)
        loss, acc, grads = lm.loss_and_grads(toks, tgts, unk=-1)
        if grads is None or not np.isfinite(loss):
            print("ARM DIVERGED at step %d" % step); print("OUTCOME: DIVERGED"); return 3
        lm.apply(grads, opt)
        if step % 250 == 0 or step == 1:
            print("step %4d/%d  loss %.4f  train acc %.2f%%  %.1fs" % (step, a.steps, loss, 100 * acc, time.time() - t0), flush=True)

    store = build_store(lm, facts, T=a.T)
    print("\nswitch: stored %d contexts (unique + the model was wrong at them)" % len(store))

    r_acc_plain, r_n, _ = score(lm, rule_te, rule_answer_start, None, a.T)
    r_acc_store, r_n2, r_hits = score(lm, rule_te, rule_answer_start, store, a.T)
    f_acc_plain, f_n, _ = score(lm, facts, fact_answer_start, None, a.T)
    f_acc_store, f_n2, f_hits = score(lm, facts, fact_answer_start, store, a.T)

    print("\n%-34s %10s %10s" % ("measurement", "model only", "with switch"))
    print("%-34s %9.2f%% %9.2f%%" % ("rule accuracy (HELD-OUT pairs)", 100 * r_acc_plain, 100 * r_acc_store))
    print("%-34s %9.2f%% %9.2f%%" % ("fact accuracy (unique keys)", 100 * f_acc_plain, 100 * f_acc_store))
    print("\n  switch hits on held-out rules: %d of %d answer tokens (collision rate %.2f%%)"
          % (r_hits, r_n, 100.0 * r_hits / max(r_n, 1)))
    print("  switch hits on facts:          %d of %d answer tokens" % (f_hits, f_n))

    p1 = f_acc_store >= 3 * f_acc_plain if f_acc_plain > 0 else f_acc_store > 0.5
    p2 = abs(r_acc_store - r_acc_plain) <= 0.01
    p3 = (r_hits / max(r_n, 1)) <= 0.01
    print("\nP1 memorisation (fact acc >= 3x model-only): %s  (%.2f%% -> %.2f%%)"
          % ("MET" if p1 else "NOT MET", 100 * f_acc_plain, 100 * f_acc_store))
    print("P2 generalisation not hurt (<= 1.0 point):  %s  (%+.2f points)"
          % ("MET" if p2 else "FAILED", 100 * (r_acc_store - r_acc_plain)))
    print("P3 no collision (<= 1%%):                    %s  (%.2f%%)" % ("MET" if p3 else "FAILED", 100.0 * r_hits / max(r_n, 1)))
    falsifier = (r_acc_store - r_acc_plain) < -0.02
    print("FALSIFIER (rule accuracy -2.0 points or worse): %s" % ("FIRED - this is a TRADE, not a free win" if falsifier else "not fired"))
    print("OUTCOME: %s" % ("SWITCH_WORKS" if (p1 and p2 and not falsifier) else "SWITCH_TRADES"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
