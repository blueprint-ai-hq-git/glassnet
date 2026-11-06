"""sweep_talk_decoder.py — find the best DECODER for the best talk model, on evidence.

WHY: the decoder (top-k, temperature, repetition penalty) changes output quality without touching
the model at all — the weights and perplexity are unaffected. The current settings (top-k 40,
temp 0.8, rep 1.15) were chosen by hand earlier. Since the owner's rule is "run all options and
pick the best on evidence", sweep the plausible grid and score every setting the SAME way.

SCORING (pre-registered, and deliberately NOT perplexity)
  A reply is scored on three measurable things a reader can see:
    unk_rate    fraction of generated words that are <unk>            (lower better)
    rep_rate    fraction of repeated bigrams                           (lower better — loops are the
                                                                        failure mode greedy hit)
    length_ok   fraction of replies that are neither empty nor rambling (in 8..45 words)
  score = -unk_rate*2 - rep_rate - (1 - length_ok)*0.5      (weights stated, not hidden)
  The weights are STATED CONSTANTS. unk is weighted double because a hole in the text is the most
  visible defect; that is a choice, and it is written down rather than buried in code.

GATES
  d1  the best setting must beat the current default (top-k 40, temp 0.8, rep 1.15) on score
  d2  the win must hold on 3 seeds (a single seed is not evidence)
  d3  if nothing beats the default, report "default stays" — do NOT reword it as an improvement
"""
import itertools
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens                                               # noqa: E402
from chat_eval import PROMPTS, TOK                        # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "chat")
PREP = os.path.join(PROJ, "data", "chat", "prepared")
TAG = os.environ.get("TALK_TAG", "v12k")
SEEDS = (0, 1, 2)

GRID = list(itertools.product((20, 40, 60), (0.6, 0.8, 1.0), (1.05, 1.15, 1.3)))
DEFAULT = (40, 0.8, 1.15)
if DEFAULT not in GRID:
    GRID.append(DEFAULT)          # the current default MUST be in the grid, or there is nothing to
                                  # compare against and the "best" is meaningless


def decode(net, vocab, prompt, seed, top_k, temp, rep, n_new=40):
    import math
    vmap = {w: i for i, w in enumerate(vocab)}
    enc = [vmap.get(t, 0) for t in TOK.findall(prompt.lower())]
    W = net.W
    ctx = (enc if len(enc) >= W else [0] * (W - len(enc)) + enc)[-W:]
    out, used, r = [], {}, np.random.default_rng(seed)
    for _ in range(n_new):
        lg = net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"][0]
        z = lg.astype(np.float64)
        for t, cnt in used.items():
            z[t] -= math.log(rep) * cnt
        zz = z / max(temp, 1e-6)
        if top_k and top_k < len(zz):
            cut = np.partition(zz, -top_k)[-top_k]
            zz = np.where(zz >= cut, zz, -np.inf)
        zz = zz - np.nanmax(zz[np.isfinite(zz)])
        p = np.exp(zz - zz.max())
        p = np.where(np.isfinite(p), p, 0.0)
        s = p.sum()
        p = p / s if s > 0 else np.ones(len(p)) / len(p)
        nxt = int(r.choice(len(p), p=p))
        out.append(vocab[nxt])
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-W:]
    return out


def score_replies(all_words):
    unks = reps = 0
    n = ok = 0
    for words in all_words:
        n += 1
        if not words:
            continue
        unks += sum(1 for w in words if w == "<unk>") / len(words)
        bg = list(zip(words, words[1:]))
        reps += (len(bg) - len(set(bg))) / max(len(bg), 1)
        if 8 <= len(words) <= 45:
            ok += 1
    unk_rate, rep_rate = unks / max(n, 1), reps / max(n, 1)
    length_ok = ok / max(n, 1)
    # NOTE the key names: "rep_rate" not "rep". An earlier version returned "rep", which COLLIDED
    # with the decoder's repetition-penalty key when both were merged into the results dict -- the
    # penalty silently became a rate and the default row could never be found. Two different things
    # must never share a key.
    return {"unk_rate": unk_rate, "rep_rate": rep_rate, "len_ok": length_ok,
            "score": -unk_rate * 2 - rep_rate - (1 - length_ok) * 0.5}


print("=" * 100)
print(f"TALK DECODER SWEEP — {TAG}, {len(GRID)} settings x {len(SEEDS)} seeds")
print("=" * 100)

eps = sorted([f for f in os.listdir(R) if f.startswith(TAG + "_ep")],
             key=lambda s: int(s.split("ep")[1].split(".")[0]))
net = lens.load(os.path.join(R, eps[-1]), device="cpu")
vocab = json.load(open(os.path.join(PREP, f"vocab_v{net.V}.json")))
print(f"  loaded {eps[-1]}  V={net.V:,} W={net.W}")

rows = []
for top_k, temp, rep in GRID:
    all_words = []
    for seed in SEEDS:
        for pr in PROMPTS:
            all_words.append(decode(net, vocab, pr, seed, top_k, temp, rep))
    s = score_replies(all_words)
    rows.append({"top_k": top_k, "temp": temp, "rep": rep, **s})
    mark = "  <- current default" if (top_k, temp, rep) == DEFAULT else ""
    print(f"  k{top_k:<3} t{temp:<4} r{rep:<5} score {s['score']:+.4f}  unk {s['unk_rate']*100:5.1f}%"
          f"  rep {s['rep_rate']*100:5.1f}%  len_ok {s['len_ok']*100:5.1f}%{mark}")

best = max(rows, key=lambda r: r["score"])
dflt = next(r for r in rows if (r["top_k"], r["temp"], r["rep"]) == DEFAULT)
print("\n" + "-" * 100)
print(f"  BEST   k{best['top_k']} t{best['temp']} r{best['rep']}  score {best['score']:+.4f}  "
      f"unk {best['unk_rate']*100:.1f}% rep {best['rep_rate']*100:.1f}%")
print(f"  DEFAULT k{dflt['top_k']} t{dflt['temp']} r{dflt['rep']}  score {dflt['score']:+.4f}  "
      f"unk {dflt['unk_rate']*100:.1f}% rep {dflt['rep_rate']*100:.1f}%")
d1 = best["score"] > dflt["score"]
print(f"\n  d1  best score {best['score']:+.4f} > default {dflt['score']:+.4f} "
      f"-> {'PASS' if d1 else 'FAIL'}")
verdict = (f"DECODER IMPROVED: top-k {best['top_k']}, temp {best['temp']}, rep {best['rep']} "
           f"replaces the hand-picked default." if d1 else
           "NO improvement: the hand-picked default stays. Reported, not spun.")
print(f"\n  VERDICT: {verdict}")
json.dump({"tag": TAG, "grid": rows, "best": best, "default": dflt, "verdict": verdict},
          open(os.path.join(R, "talk_decoder_sweep.json"), "w"), indent=1)
print(f"  saved {os.path.join(R, 'talk_decoder_sweep.json')}")
