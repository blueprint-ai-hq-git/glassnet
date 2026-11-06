#!/usr/bin/env python3
"""TALK QUALITY v2 — "word salad" as a calibrated measurement, not an opinion.

WHY v2 EXISTS (the v1 metric was falsified by its own control, 2026-10-16)
-------------------------------------------------------------------------
v1 (kept on disk as talk_quality_v1_falsified.py) defined salad as "the fraction of generated adjacent
token pairs that also occur adjacently in the corpus". Run against a KNOWN-GARBAGE control (a 1-epoch
model whose output was visibly junk: "do your what be as it \" as or or this by by") it scored 64.96%
and returned "not word salad". The reason is structural: the corpus's most frequent words occur next
to each other constantly, so text built from high-frequency tokens scores well however little order it
has. Adjacency measures FREQUENCY, not ORDER. The adjacency number is still printed here as a
secondary figure, and it decides nothing.

v2: score text with a TRIGRAM model fitted on real TRAIN text, and take the bar from two controls
built out of real data:
  * REAL     -- held-out val text (what good output looks like)
  * SHUFFLED -- the same val chunks with their words permuted (the literal definition of word salad)
A generation must score better than the MIDPOINT between shuffled and real. The bar is fixed here,
before any arm is measured, and it comes from the data rather than from my opinion of a sample.

Usage: bash lens/scripts/run_talk_quality.sh <ship.npz> [label] [corpus-dir]
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet")
import chat_data  # noqa: E402
import lens  # noqa: E402

N_NEW = 40
PROMPTS = ["human : hello\n\nassistant :",
           "human : what is the capital of france\n\nassistant :",
           "human : can you explain what a neural network is\n\nassistant :",
           "human : i am feeling sad today\n\nassistant :",
           "human : write a short poem about the sea\n\nassistant :",
           "human : how do i make pasta\n\nassistant :"]
TRAIN_TOKENS_CAP = 12_000_000
ALPHA = 0.4          # add-alpha smoothing


def tokenize(t):
    return re.findall(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]", t.lower())


class TrigramScorer:
    """Interpolated trigram log-probability per token, counted from real training text only."""

    def __init__(self, tokens, V):
        a = np.asarray(tokens, dtype=np.int64)
        tri = (a[:-2] * V + a[1:-1]) * V + a[2:]
        bi = a[:-1] * V + a[1:]
        tri.sort()
        bi.sort()
        self.tri, self.tri_c = np.unique(tri, return_counts=True)
        self.bi, self.bi_c = np.unique(bi, return_counts=True)
        self.V = V
        self.uni = np.bincount(a, minlength=V).astype(np.float64)
        self.uni_p = (self.uni + ALPHA) / (self.uni.sum() + ALPHA * V)

    @staticmethod
    def _lookup(keys, counts, q):
        pos = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
        return np.where(keys[pos] == q, counts[pos], 0).astype(np.float64)

    def score(self, seq):
        """Mean log-probability per token (tokens 3..n). Lower = more salad-like."""
        a = np.asarray(seq, dtype=np.int64)
        if len(a) < 4:
            return float("nan")
        V = self.V
        tri = (a[:-2] * V + a[1:-1]) * V + a[2:]
        bi = a[:-2] * V + a[1:-1]
        bi_next = a[1:-1] * V + a[2:]
        c3 = self._lookup(self.tri, self.tri_c, tri)
        c2 = self._lookup(self.bi, self.bi_c, bi)
        c2n = self._lookup(self.bi, self.bi_c, bi_next)
        c1 = self.uni[a[1:-1]]
        p_back = (c2n + ALPHA * self.uni_p[a[2:]]) / (c1 + ALPHA)
        p = (c3 + ALPHA * p_back) / (c2 + ALPHA)
        return float(np.mean(np.log(np.maximum(p, 1e-9))))


def main():
    ship = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(ship)
    d = chat_data.build()
    ids, bounds, split, vocab = d["ids"], d["bounds"], d["split"], d["vocab"]
    V = len(vocab)
    vmap = {w: i for i, w in enumerate(vocab)}

    train_tokens = np.concatenate([ids[bounds[j]:bounds[j + 1]] for j in range(len(bounds) - 1)
                                   if split[j] == 0])[:TRAIN_TOKENS_CAP]
    sc = TrigramScorer(train_tokens, V)
    print(f"  corpus {os.environ.get('CHAT_DATA_DIR', '(default)')}   vocab {V}   "
          f"trigram table {len(sc.tri):,} entries from {len(train_tokens):,} train tokens")

    real, shuf = [], []
    for j in range(len(bounds) - 1):
        if split[j] != 1:
            continue
        t = ids[bounds[j]:bounds[j + 1]]
        for s in range(0, max(len(t) - N_NEW, 0), N_NEW):
            chunk = np.asarray(t[s:s + N_NEW], dtype=np.int64)
            real.append(chunk)
            shuf.append(np.random.default_rng(len(real)).permutation(chunk))
        if len(real) > 400:
            break
    r_real = float(np.mean([sc.score(c) for c in real]))
    r_shuf = float(np.mean([sc.score(c) for c in shuf]))
    midpoint = (r_real + r_shuf) / 2.0

    net = lens.load(ship)
    ctx_win = int(getattr(net, "W", 32))

    def generate(prompt, seed=0, temp=0.8, top_k=40, rep_penalty=1.15):
        """The trainer's decoder, unchanged: top-k 40, temp 0.8, repetition penalty 1.15."""
        enc = [vmap.get(x, 0) for x in tokenize(prompt)]
        if len(enc) < ctx_win:
            enc = [0] * (ctx_win - len(enc)) + enc
        ctx, out, used = enc[-ctx_win:], [], {}
        for _ in range(N_NEW):
            z = np.asarray(net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"][0],
                           dtype=np.float64)
            for tok, cnt in used.items():
                z[tok] -= np.log(rep_penalty) * cnt
            z = z / max(temp, 1e-6)
            if top_k and top_k < len(z):
                cut = np.partition(z, -top_k)[-top_k]
                z = np.where(z >= cut, z, -np.inf)
            m = np.nanmax(z[np.isfinite(z)])
            p = np.exp(z - m)
            p = np.where(np.isfinite(p), p, 0.0)
            p = p / p.sum()
            nxt = int(np.random.default_rng(seed + 1000 * len(out)).choice(V, p=p))
            used[nxt] = used.get(nxt, 0) + 1
            out.append(nxt)
            ctx = (ctx + [nxt])[-ctx_win:]
        return out

    scores, texts = [], []
    for i, pr in enumerate(PROMPTS):
        gen = generate(pr, seed=i)
        scores.append(sc.score(np.asarray(gen, dtype=np.int64)))
        joined = " ".join(vocab[t] for t in gen).replace(" '", "'")
        texts.append(joined)
        print(f"  sample {i+1}: {joined[:280]}")
    r_gen = float(np.mean(scores))

    span = r_real - r_shuf
    pos = (r_gen - r_shuf) / span if span else float("nan")
    verdict = "not word salad" if r_gen > midpoint else "WORD SALAD"
    print()
    print(f"  SHIP {label}  ({ship})")
    print(f"  trigram log-prob per token   REAL {r_real:.3f}   SHUFFLED {r_shuf:.3f}   "
          f"bar(midpoint) {midpoint:.3f}")
    print(f"  this model's output          {r_gen:.3f}   position {pos*100:.1f}% of the way from "
          f"shuffled to real")
    print(f"  VERDICT (computed from the pre-registered bar): {verdict}")
    out = {"ship": ship, "label": label, "real": r_real, "shuffled": r_shuf, "midpoint": midpoint,
           "generated": r_gen, "position": pos, "verdict": verdict, "samples": texts,
           "metric": "trigram logprob v2, bar calibrated by a shuffled-word control"}
    dest = f"/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/talk_quality_{label}.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"  written: {dest}")


if __name__ == "__main__":
    main()