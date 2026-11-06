#!/usr/bin/env python3
"""TALK QUALITY — turns "word salad" from an opinion into a measured number.

DEFINITION (pre-registered 2026-10-16, before any arm was run)
--------------------------------------------------------------
Real language reuses the same adjacent word pairs over and over. Word salad does not: it emits
token pairs that never occur next to each other in the corpus. So:

  adjacency rate = fraction of adjacent token pairs in a text that are pairs the TRAINING corpus
                   also contains adjacently.

The meter measures that rate on (a) the model's generated replies and (b) real HELD-OUT val text.
Held-out real text is the reference ceiling, so the comparison carries its own control: if the
model scores far below real text, it is producing token pairs that do not occur in English-ish
dialogue, which is what "word salad" means.

BAR (fixed before the run): generated adjacency rate < 50% of the real held-out rate => WORD SALAD.
The threshold is stated here so it cannot be moved after seeing the result.
Also reported: distinct-1 / distinct-2 (lexical variety), the longest run of one repeated token,
and the raw samples — because a number without the text it describes is not evidence.

Usage:  bash lens/scripts/run_talk_quality.sh <ship.npz> [label]
Corpus comes from CHAT_DATA_DIR (default data/chat/prepared_ext).
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

WIN = 32
N_NEW = 40
SALAD_RATIO = 0.50           # pre-registered bar
PROMPTS = ["human : hello\n\nassistant :",
           "human : what is the capital of france\n\nassistant :",
           "human : can you explain what a neural network is\n\nassistant :",
           "human : i am feeling sad today\n\nassistant :",
           "human : write a short poem about the sea\n\nassistant :",
           "human : how do i make pasta\n\nassistant :"]
PAIR_SAMPLE = 12_000_000     # cap on train pairs held in memory (~96 MB as int64)


def tokenize(t):
    return re.findall(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]", t.lower())


def pair_hashes(seq, V):
    a = np.asarray(seq, dtype=np.int64)
    return a[:-1] * V + a[1:]


def main():
    ship = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(ship)
    d = chat_data.build()
    ids, bounds, split, vocab = d["ids"], d["bounds"], d["split"], d["vocab"]
    V = len(vocab)
    vmap = {w: i for i, w in enumerate(vocab)}

    train_tokens = np.concatenate([ids[bounds[j]:bounds[j + 1]] for j in range(len(bounds) - 1)
                                   if split[j] == 0])
    pairs = pair_hashes(train_tokens[:PAIR_SAMPLE], V)
    pairs.sort()
    print(f"  corpus {os.environ.get('CHAT_DATA_DIR', '(default)')}   vocab {V}   "
          f"train pairs indexed {len(pairs):,}")

    def adjacency(seq):
        if len(seq) < 2:
            return float("nan")
        q = pair_hashes(seq, V)
        return float(np.isin(q, pairs).mean())

    # reference: real HELD-OUT text, in chunks the same length as a generated reply
    real_rates, real_tokens = [], []
    for j in range(len(bounds) - 1):
        if split[j] != 1:
            continue
        t = ids[bounds[j]:bounds[j + 1]]
        real_tokens.extend(t.tolist())
        for s in range(0, max(len(t) - N_NEW, 0), N_NEW):
            real_rates.append(adjacency(t[s:s + N_NEW]))
        if len(real_rates) > 400:
            break
    real_rate = float(np.mean(real_rates))

    net = lens.load(ship)
    ctx_win = int(getattr(net, "W", WIN))   # the model's OWN window, so the meter matches the ship

    def encode(t):
        return [vmap.get(x, 0) for x in tokenize(t)]

    def generate(prompt, seed=0, temp=0.8, top_k=40, rep_penalty=1.15):
        """The trainer's decoder, unchanged: top-k 40, temp 0.8, repetition penalty 1.15."""
        enc = encode(prompt)
        if len(enc) < ctx_win:
            enc = [0] * (ctx_win - len(enc)) + enc
        ctx = enc[-ctx_win:]
        out, used = [], {}
        for _ in range(N_NEW):
            # forward_arrays returns host NumPy (no cupy/numpy mixing), logits for the one window
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

    rates, texts = [], []
    for i, pr in enumerate(PROMPTS):
        gen = generate(pr, seed=i)
        rates.append(adjacency(np.asarray(gen)))
        toks = [vocab[t] for t in gen]
        joined = " ".join(toks).replace(" '", "'")
        texts.append(joined)
        print(f"  sample {i+1}: {joined[:300]}")
    gen_rate = float(np.mean(rates))

    all_gen = [t for txt in texts for t in tokenize(txt)]
    d1 = len(set(all_gen)) / max(len(all_gen), 1)
    bigrams = list(zip(all_gen[:-1], all_gen[1:]))
    d2 = len(set(bigrams)) / max(len(bigrams), 1)
    longest, cur = 1, 1
    for a, b in zip(all_gen, all_gen[1:]):
        cur = cur + 1 if a == b else 1
        longest = max(longest, cur)

    ratio = gen_rate / real_rate if real_rate else float("nan")
    print()
    print(f"  SHIP {label}  ({ship})")
    print(f"  adjacency rate   generated {gen_rate*100:.2f}%   real held-out {real_rate*100:.2f}%   "
          f"ratio {ratio:.2f}")
    print(f"  distinct-1 {d1:.3f}  distinct-2 {d2:.3f}  longest repeated-token run {longest}")
    print(f"  BAR (pre-registered): ratio < {SALAD_RATIO:.2f} => WORD SALAD.  "
          f"VERDICT: {'WORD SALAD' if ratio < SALAD_RATIO else 'not word salad'}")
    out = {"ship": ship, "label": label, "gen_adjacency": gen_rate, "real_adjacency": real_rate,
           "ratio": ratio, "distinct1": d1, "distinct2": d2, "longest_run": longest,
           "bar": SALAD_RATIO, "verdict": "WORD SALAD" if ratio < SALAD_RATIO else "not word salad",
           "samples": texts}
    dest = f"/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/talk_quality_{label}.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"  written: {dest}")


if __name__ == "__main__":
    main()