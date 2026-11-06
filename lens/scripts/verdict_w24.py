"""verdict_w24.py — gates for the context-length experiment (W=24 vs W=16, both V=12000).

Gates (fixed BEFORE the run in run_w24.sh):
  w1  val_ppl at W=24 beats v12k (W=16, same vocab) by > 3%
  w2  bits/token improves by > 0.05  (not just raw ppl — must be per-token)
  w3  if w1 fails, report and stop adding context
Same prompts, same decoder, same <unk> measurement as every other verdict in this project.
"""
import json
import math
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens                                              # noqa: E402
from chat_eval import PROMPTS, decode                    # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "chat")
PREP = os.path.join(PROJ, "data", "chat", "prepared")


def evaluate(tag):
    eps = sorted([f for f in os.listdir(R) if f.startswith(tag + "_ep")],
                 key=lambda s: int(s.split("ep")[1].split(".")[0]))
    if not eps:
        return None
    net = lens.load(os.path.join(R, eps[-1]), device="cpu")
    vfile = os.path.join(PREP, f"vocab_v{net.V}.json")
    if not os.path.exists(vfile):
        return None
    vocab = json.load(open(vfile))
    if len(vocab) != net.V:
        return None
    log = json.load(open(os.path.join(R, f"{tag}_log.json")))
    ppl = log["history"][-1]["val_ppl"]
    unks, samples = [], []
    for i, pr in enumerate(PROMPTS):
        words = decode(net, vocab, pr, seed=i)
        unks.append(sum(1 for w in words if w == "<unk>") / len(words))
        if i < 3:
            samples.append(" ".join(words))
    return {"tag": tag, "V": len(vocab), "W": net.W, "ppl": ppl, "bpt": math.log2(ppl),
            "unk": float(np.mean(unks)), "samples": samples}


print("=" * 100)
print("VERDICT: W=24 vs W=16 (both V=12000, same recipe) — does longer context help?")
print("=" * 100)

base = evaluate("v12k")
new = evaluate("w24_v12000")

if new is None:
    print("  w24_v12000 not finished — verdict withheld")
    sys.exit(0)

print(f"\n  {'model':<14}{'V':>8}{'W':>4}{'val_ppl':>10}{'bits/token':>12}{'<unk>':>8}")
for m in (base, new):
    if m:
        print(f"  {m['tag']:<14}{m['V']:>8,}{m['W']:>4}{m['ppl']:>10.2f}{m['bpt']:>12.2f}"
              f"{m['unk']*100:>7.1f}%")

print("\n  SAMPLE REPLIES (W=24):")
for s in new["samples"]:
    print(f"    > {s[:94]}")

if base:
    d_ppl = (base["ppl"] - new["ppl"]) / base["ppl"] * 100
    d_bpt = new["bpt"] - base["bpt"]
    d_unk = (base["unk"] - new["unk"]) * 100
    w1 = d_ppl > 3.0
    w2 = d_bpt < -0.05
    print("\n" + "-" * 100)
    print("  GATES — every compared value beside its verdict")
    print(f"    w1  ppl {new['ppl']:.2f} vs {base['ppl']:.2f}  improvement {d_ppl:+.1f}% (bar >3%) "
          f"-> {'PASS' if w1 else 'FAIL'}")
    print(f"    w2  bits/token {new['bpt']:.2f} vs {base['bpt']:.2f}  delta {d_bpt:+.3f} "
          f"(bar -0.05) -> {'PASS' if w2 else 'FAIL'}")
    print(f"    (context for curiosity: <unk> {base['unk']*100:.1f}% -> {new['unk']*100:.1f}% "
          f"= {d_unk:+.1f} pts)")
    if w1 and w2:
        verdict = ("W=24 WINS: longer context helps on both metrics -> W=24 becomes the context "
                   "default for the talk model.")
    else:
        verdict = ("W=24 FAILS the pre-registered gates: longer context does not help this model at "
                   "this data size. Report and STOP adding context — do not keep stacking changes "
                   "hoping something moves.")
    print(f"\n  VERDICT: {verdict}")
    json.dump({"base": {k: v for k, v in base.items() if k != "samples"},
               "new": {k: v for k, v in new.items() if k != "samples"},
               "gates": {"w1": bool(w1), "w2": bool(w2)}, "verdict": verdict},
              open(os.path.join(R, "w24_verdict.json"), "w"), indent=1)
    print(f"  saved {os.path.join(R, 'w24_verdict.json')}")
