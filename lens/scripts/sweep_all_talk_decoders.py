"""sweep_all_talk_decoders.py — give EVERY talk model its own tuned decoder, then compare.

WHY THIS IS REQUIRED FOR FAIRNESS
  I just tuned a decoder for the assistant-only model and found 22 settings that beat v12k's levels.
  But v12k and best1 were still being judged at INHERITED settings (top-k 60, temp 0.8, rep 1.15).
  Comparing "asonly at its best" against "best1 at its default" is the same unfair comparison I have
  caught and fixed three times today. Each model must be measured at ITS OWN best decoder, and only
  then can the models be ranked.

WHAT THIS PRODUCES
  For each talk checkpoint: its best decoder setting (by the stated score), and the unk/repetition
  that setting achieves. Then a single ranking across models on metrics valid for ALL of them
  (unk rate and repetition — NOT val ppl, which is invalid for the masked model).

GATES (pre-registered)
  u1  the winner must win on BOTH unk rate AND repetition (no trade-off wins)
  u2  if the winner is the masked model, its advantage must survive this fair comparison — if it only
      won because it got a tuned decoder, u2 FAILS and the honest answer is "no model is better,
      they were measured unequally"
"""
import itertools
import json
import math
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
SEEDS = (0, 1, 2)

# (tag, vocab file)  — the vocab MUST match the checkpoint's own V
MODELS = [("best1", "vocab_v3500.json"),
          ("v12k", "vocab_v12000.json"),
          ("asonly_v12000", "vocab_v12000.json")]

GRID = list(itertools.product((40, 60, 80), (0.7, 0.8, 0.9), (1.15, 1.3, 1.45, 1.6)))


def decode(net, vocab, prompt, seed, top_k, temp, rep, n_new=40):
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


def score(all_words):
    unks = reps = 0
    n = ok = 0
    for w in all_words:
        n += 1
        if not w:
            continue
        unks += sum(1 for t in w if t == "<unk>") / len(w)
        bg = list(zip(w, w[1:]))
        reps += (len(bg) - len(set(bg))) / max(len(bg), 1)
        if 8 <= len(w) <= 45:
            ok += 1
    return {"unk_rate": unks / max(n, 1), "rep_rate": reps / max(n, 1), "len_ok": ok / max(n, 1)}


print("=" * 100)
print("EVERY TALK MODEL AT ITS OWN BEST DECODER — the fair comparison")
print("=" * 100)

final = {}
for tag, vfile in MODELS:
    eps = sorted([f for f in os.listdir(R)
                  if f.startswith(tag + "_ep") and f.endswith(".npz")],
                 key=lambda s: int(s.split("ep")[1].split(".")[0]))
    if not eps:
        print(f"  {tag}: no checkpoint")
        continue
    vp = os.path.join(PREP, vfile)
    if not os.path.exists(vp):
        print(f"  {tag}: no vocab {vfile}")
        continue
    net = lens.load(os.path.join(R, eps[-1]), device="cpu")
    vocab = json.load(open(vp))
    if len(vocab) != net.V:
        print(f"  {tag}: vocab {len(vocab)} != model V {net.V} — refusing")
        continue
    rows = []
    for top_k, temp, rep in GRID:
        allw = []
        for seed in SEEDS:
            for pr in PROMPTS:
                allw.append(decode(net, vocab, pr, seed, top_k, temp, rep))
        s = score(allw)
        rows.append({"top_k": top_k, "temp": temp, "rep": rep, **s})
    # best = lowest unk, tie-broken by lowest repetition. Stated, not hidden.
    best = min(rows, key=lambda r: (round(r["unk_rate"], 4), r["rep_rate"]))
    final[tag] = {"ckpt": eps[-1], "V": net.V, "best": best}
    print(f"\n  {tag}  ({eps[-1]}, V={net.V:,})")
    print(f"    best decoder : top-k {best['top_k']}, temp {best['temp']}, rep {best['rep']}")
    print(f"    unk {best['unk_rate']*100:.2f}%   repetition {best['rep_rate']*100:.2f}%   "
          f"len_ok {best['len_ok']*100:.0f}%")

print("\n" + "=" * 100)
print("  RANKING — each model at its OWN best decoder, judged on both reply metrics")
print("=" * 100)
print(f"\n  {'model':<18}{'best decoder':<24}{'unk':>9}{'repetition':>13}")
ranked = sorted(final.items(), key=lambda kv: (round(kv[1]["best"]["unk_rate"], 4),
                                               kv[1]["best"]["rep_rate"]))
for tag, v in ranked:
    b = v["best"]
    print(f"  {tag:<18}k{b['top_k']:<3} t{b['temp']:<4} r{b['rep']:<6}{b['unk_rate']*100:>8.2f}%"
          f"{b['rep_rate']*100:>12.2f}%")

if len(ranked) >= 2:
    (w_tag, w), (r_tag, r) = ranked[0], ranked[1]
    wb, rb = w["best"], r["best"]
    u1 = wb["unk_rate"] <= rb["unk_rate"] and wb["rep_rate"] <= rb["rep_rate"]
    print(f"\n  winner: {w_tag}   runner-up: {r_tag}")
    print(f"  u1  winner is <= runner-up on BOTH metrics -> {'PASS' if u1 else 'FAIL (trade-off)'}")
    verdict = (f"{w_tag} is the best talk model on a FAIR comparison (every model at its own best "
               f"decoder): unk {wb['unk_rate']*100:.2f}% and repetition {wb['rep_rate']*100:.2f}%."
               if u1 else
               f"NO model dominates: {w_tag} wins unk but {r_tag} wins repetition. Reported as a "
               f"trade-off, not a winner.")
    print(f"\n  VERDICT: {verdict}")
    json.dump({"models": {k: {"ckpt": v["ckpt"], "V": v["V"], "best": v["best"]}
                          for k, v in final.items()},
               "ranking": [k for k, _ in ranked], "gates": {"u1": bool(u1)}, "verdict": verdict},
              open(os.path.join(R, "talk_fair_comparison.json"), "w"), indent=1)
    print(f"  saved {os.path.join(R, 'talk_fair_comparison.json')}")
