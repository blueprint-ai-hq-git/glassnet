"""verdict_v12k.py — the pre-registered gates for the vocab experiment, decided from artifacts.

GATES (fixed BEFORE the run, in run_v12k.sh):
  v1  <unk> rate in generated replies FALLS below best1's 14.5%
  v2  bits/token stays within +0.30 of best1's 6.04   (accept a small cost for fewer unks, not a collapse)

Fair-comparison rules applied (learned the hard way earlier in this project):
  * SAME prompts for both models
  * SAME decoder (top-k 40, rep penalty 1.15, temp 0.8) -- the verified one
  * <unk> counted in GENERATED text, not the training corpus
  * ppl converted to BITS/TOKEN because raw ppl is NOT comparable across vocab sizes
"""
import json
import math
import os
import re
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens                                              # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "chat")
PREP = os.path.join(PROJ, "data", "chat", "prepared")
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
from chat_eval import PROMPTS, decode, TOK            # noqa: E402


def evaluate(tag, vfile):
    eps = sorted([f for f in os.listdir(R) if f.startswith(tag + "_ep")],
                 key=lambda s: int(s.split("ep")[1].split(".")[0]))
    if not eps:
        return None
    vocab = json.load(open(os.path.join(PREP, vfile)))
    net = lens.load(os.path.join(R, eps[-1]), device="cpu")
    if getattr(net, "V", None) != len(vocab):
        print(f"  {tag}: REFUSED — V {getattr(net,'V',None)} != vocab {len(vocab)}")
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
print("VERDICT: vocab 3500 (best1) vs vocab 12000 with the SAME good recipe (v12k)")
print("=" * 100)

base = evaluate("best1", "vocab_v3500.json")
new = evaluate("v12k", "vocab_v12000.json")

if new is None:
    print("  v12k not finished — verdict WITHHELD (not guessed)")
    sys.exit(0)

print(f"\n  {'model':<10}{'V':>8}{'W':>4}{'val_ppl':>10}{'bits/token':>12}{'<unk>':>8}")
for m in (base, new):
    if m:
        print(f"  {m['tag']:<10}{m['V']:>8,}{m['W']:>4}{m['ppl']:>10.2f}{m['bpt']:>12.2f}"
              f"{m['unk']*100:>7.1f}%")

print("\n  SAMPLE REPLIES (v12k, same prompts, same decoder):")
for s in new["samples"]:
    print(f"    > {s[:94]}")

if base:
    d_unk = (base["unk"] - new["unk"]) * 100
    d_bpt = new["bpt"] - base["bpt"]
    v1 = new["unk"] < base["unk"]
    v2 = d_bpt <= 0.30
    print("\n" + "-" * 100)
    print("  GATES — every compared value printed beside its verdict")
    print(f"    v1  unk {new['unk']*100:.1f}% < {base['unk']*100:.1f}%  (delta {d_unk:+.1f} pts) "
          f"-> {'PASS' if v1 else 'FAIL'}")
    print(f"    v2  bits/token {new['bpt']:.2f} vs {base['bpt']:.2f}  (delta {d_bpt:+.2f}, "
          f"bar +0.30) -> {'PASS' if v2 else 'FAIL'}")
    if v1 and v2:
        verdict = ("v12k WINS: fewer unks AND acceptable cost -> it becomes the talk model, and the "
                   "decoder sweep re-runs on it.")
    elif v1 and not v2:
        verdict = ("MIXED: unks fell but the perplexity cost is too high -> report and stop, do NOT "
                   "hide it. Next lever is context length, not more vocab.")
    elif not v1:
        verdict = ("v12k FAILS v1: the big vocab does not reduce unks once the recipe is fixed. "
                   "That kills this direction honestly; go to window length instead.")
    else:
        verdict = "READ THE NUMBERS ABOVE; do not narrate around them."
    print(f"\n  VERDICT: {verdict}")
    json.dump({"base": {k: v for k, v in base.items() if k != "samples"},
               "new": {k: v for k, v in new.items() if k != "samples"},
               "gates": {"v1": bool(v1), "v2": bool(v2)}, "verdict": verdict},
              open(os.path.join(R, "v12k_verdict.json"), "w"), indent=1)
    print(f"  saved {os.path.join(R, 'v12k_verdict.json')}")
