"""sweep_asonly_decoder.py — tune the decoder FOR the assistant-only model.

WHY THIS IS THE RIGHT NEXT STEP
  The assistant-only model reached unk 0.0% (zero holes — best in the project) but repetition rose
  from 1.8% to 4.7%. That regression was measured with the decoder tuned on a DIFFERENT model (v12k's
  top-k 60). A decoder is part of the system, so when the model changes the decoder should be
  re-tuned for it. That was never done, and it is cheap (generation only, no training).

THE QUESTION
  Can a decoder tuned for this model keep unk at ~0% while bringing repetition back under v12k's
  1.8%? If yes, the MIXED verdict becomes a genuine reply-quality win. If no, the repetition is
  intrinsic to the masked model and that is the finding.

WIDER REPETITION RANGE: the default grid went to rep 1.3. Repetition is this model's specific
weakness, so the range is extended to 1.6 to give the fix a real chance — and if even 1.6 cannot
fix it, the weakness is intrinsic rather than a tuning miss.

GATES (pre-registered)
  t1  some setting reaches unk <= 1.0% AND repetition < 1.8% (v12k's level) -> MIXED becomes a WIN
  t2  if no setting reaches both, report that the repetition is intrinsic to the masked model and
      keep the honest MIXED verdict. Do NOT relax the gate to manufacture a pass.
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
TAG = "asonly_v12000"
SEEDS = (0, 1, 2)
V12K_UNK, V12K_REP = 0.031, 0.018          # measured baseline levels

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
print(f"ASONLY DECODER SWEEP — {TAG}, {len(GRID)} settings x {len(SEEDS)} seeds")
print(f"  target: unk <= 1.0% AND repetition < {V12K_REP*100:.1f}% (v12k's level)")
print("=" * 100)

eps = sorted([f for f in os.listdir(R)
                  if f.startswith(TAG + "_ep") and f.endswith(".npz")],
             key=lambda s: int(s.split("ep")[1].split(".")[0]))
net = lens.load(os.path.join(R, eps[-1]), device="cpu")
vocab = json.load(open(os.path.join(PREP, f"vocab_v{net.V}.json")))
print(f"  loaded {eps[-1]}  V={net.V:,} W={net.W}\n")

rows = []
for top_k, temp, rep in GRID:
    allw = []
    for seed in SEEDS:
        for pr in PROMPTS:
            allw.append(decode(net, vocab, pr, seed, top_k, temp, rep))
    s = score(allw)
    rows.append({"top_k": top_k, "temp": temp, "rep": rep, **s})
    good = s["unk_rate"] <= 0.01 and s["rep_rate"] < V12K_REP
    print(f"  k{top_k:<3} t{temp:<4} r{rep:<5} unk {s['unk_rate']*100:5.2f}%  "
          f"rep {s['rep_rate']*100:5.2f}%  len_ok {s['len_ok']*100:5.1f}%"
          f"{'   <- MEETS BOTH' if good else ''}")

meet = [r for r in rows if r["unk_rate"] <= 0.01 and r["rep_rate"] < V12K_REP]
print("\n" + "-" * 100)
if meet:
    best = min(meet, key=lambda r: r["rep_rate"])
    t1 = True
    print(f"  {len(meet)} setting(s) meet BOTH targets. Best: k{best['top_k']} t{best['temp']} "
          f"r{best['rep']}")
    print(f"    unk {best['unk_rate']*100:.2f}% (<=1.0% target), rep {best['rep_rate']*100:.2f}% "
          f"(<{V12K_REP*100:.1f}% target)")
    verdict = (f"ASONLY DECODER TUNED: k{best['top_k']} t{best['temp']} r{best['rep']} keeps unk at "
               f"{best['unk_rate']*100:.2f}% while cutting repetition to {best['rep_rate']*100:.2f}%. "
               f"The MIXED verdict becomes a reply-quality WIN: zero holes AND less looping than "
               f"the baseline.")
else:
    t1 = False
    best = min(rows, key=lambda r: (r["rep_rate"], r["unk_rate"]))
    print(f"  NO setting meets both. Closest: k{best['top_k']} t{best['temp']} r{best['rep']}")
    print(f"    unk {best['unk_rate']*100:.2f}%  rep {best['rep_rate']*100:.2f}%")
    verdict = (f"NO SETTING MEETS BOTH TARGETS even at rep penalty 1.6 (highest tested). The "
               f"repetition is INTRINSIC to the assistant-only model, not a decoder-tuning miss. "
               f"The honest MIXED verdict stands: zero holes, but the model loops more, and no "
               f"decoder setting fixes that.")
print(f"\n  t1  some setting reaches unk<=1% AND rep<1.8% -> {'PASS' if t1 else 'FAIL'}")
print(f"\n  VERDICT: {verdict}")
json.dump({"tag": TAG, "grid": rows, "target": {"unk": 0.01, "rep": V12K_REP},
           "gates": {"t1": bool(t1)}, "verdict": verdict},
          open(os.path.join(R, "asonly_decoder_sweep.json"), "w"), indent=1)
print(f"  saved {os.path.join(R, 'asonly_decoder_sweep.json')}")
