"""diag_assistant_ppl.py — is the masked run FAILING, or is my METRIC wrong?

THE SYMPTOM: the assistant-only run shows val ppl 6425 -> 7480 and RISING, while v12k reached 98.
That looks like divergence. But val ppl is computed over ALL tokens, including the HUMAN's text --
and the mask deliberately stops training on human text. So the model is being graded on exactly what
it was told to ignore.

THE TEST: recompute validation loss SEPARATELY on assistant targets and human targets for both
models. If the masked model is better on assistant text and worse on human text, the mask is working
and the metric was misleading. If it is worse on BOTH, the run genuinely failed.

This decides which story is true, from artifacts, without re-running training.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import chat_data                                          # noqa: E402
import lens                                               # noqa: E402
from lens import backend as B                             # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "chat")

d = chat_data.build()
ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]

# build the assistant flag over the whole corpus (same walk as training)
is_asst = np.zeros(len(ids), dtype=np.float32)
in_asst = False
for j in range(len(bounds) - 1):
    a0, b0 = bounds[j], bounds[j + 1]
    toks = [vocab[i] for i in ids[a0:b0]]
    k = 0
    while k < len(toks):
        if toks[k].lower() == "assistant" and k + 1 < len(toks) and toks[k + 1] == ":":
            in_asst = True; k += 2; continue
        if toks[k].lower() == "human" and k + 1 < len(toks) and toks[k + 1] == ":":
            in_asst = False; k += 2; continue
        if in_asst:
            is_asst[a0 + k] = 1.0
        k += 1

Xva, Yva = chat_data.make_windows(ids, bounds, split, 16, which=1)
Wva = is_asst[Yva]
print(f"  validation windows {len(Xva):,}  assistant targets {int(Wva.sum()):,} "
      f"({Wva.mean()*100:.1f}%)")


def split_ce(tag):
    eps = sorted([f for f in os.listdir(R) if f.startswith(tag + "_ep")],
                 key=lambda s: int(s.split("ep")[1].split(".")[0]))
    if not eps:
        return None
    net = lens.load(os.path.join(R, eps[-1]), device="cpu")
    n = min(len(Xva), 20000)
    X, Y, W = Xva[:n], Yva[:n], Wva[:n]
    tot_a, cnt_a, tot_h, cnt_h = 0.0, 0, 0.0, 0
    import numpy as _np
    for s in range(0, n, 512):
        xb, yb, wb = X[s:s + 512], Y[s:s + 512], W[s:s + 512]
        lg = B.to_cpu(net.forward_arrays(xb)["logits"]).astype(np.float64)
        z = lg - lg.max(axis=1, keepdims=True)
        p = _np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        nll = -_np.log(p[_np.arange(len(yb)), yb] + 1e-12)
        a = wb > 0.5
        if a.any():
            tot_a += float(nll[a].sum()); cnt_a += int(a.sum())
        if (~a).any():
            tot_h += float(nll[~a].sum()); cnt_h += int((~a).sum())
    import math
    ce_a = tot_a / max(cnt_a, 1)
    ce_h = tot_h / max(cnt_h, 1)
    return {"tag": tag, "ckpt": eps[-1], "ce_asst": ce_a, "ce_human": ce_h,
            "ppl_asst": math.exp(min(ce_a, 30)), "ppl_human": math.exp(min(ce_h, 30)),
            "ce_all": (tot_a + tot_h) / max(cnt_a + cnt_h, 1),
            "ppl_all": math.exp(min((tot_a + tot_h) / max(cnt_a + cnt_h, 1), 30))}


print("=" * 100)
print("ASSISTANT vs HUMAN VALIDATION LOSS — is the masked run failing, or is the metric wrong?")
print("=" * 100)
res = [r for r in (split_ce("v12k"), split_ce("asonly_v12000")) if r]
for r in res:
    print(f"\n  {r['tag']}  ({r['ckpt']})")
    print(f"    ALL tokens    ppl {r['ppl_all']:>10.2f}")
    print(f"    ASSISTANT     ppl {r['ppl_asst']:>10.2f}   (the turn the model is judged on)")
    print(f"    HUMAN         ppl {r['ppl_human']:>10.2f}   (deliberately not trained on)")

if len(res) == 2:
    a, b = res[0], res[1]
    print("\n" + "-" * 100)
    print(f"  assistant ppl : {a['tag']} {a['ppl_asst']:.2f}  ->  {b['tag']} {b['ppl_asst']:.2f}")
    print(f"  human     ppl : {a['tag']} {a['ppl_human']:.2f}  ->  {b['tag']} {b['ppl_human']:.2f}")
    better_asst = b["ppl_asst"] < a["ppl_asst"]
    worse_human = b["ppl_human"] > a["ppl_human"]
    print(f"\n  mask better on ASSISTANT text : {better_asst}")
    print(f"  mask worse on HUMAN text      : {worse_human}")
    if better_asst and worse_human:
        v = ("THE MASK IS WORKING and the metric was misleading: it improves the assistant's own "
             "turn while getting worse at re-predicting the human's — exactly the trade the mask "
             "asks for. Overall ppl is the WRONG metric for this model.")
    elif not better_asst:
        v = ("THE MASK IS NOT WORKING: it is worse on assistant text too, so the run genuinely "
             "failed. Do not claim otherwise.")
    else:
        v = "READ THE NUMBERS ABOVE."
    print(f"\n  VERDICT: {v}")
    json.dump({"v12k": a, "asonly": b, "mask_helps_assistant": bool(better_asst),
               "verdict": v}, open(os.path.join(R, "assistant_ppl_diag.json"), "w"), indent=1)
    print(f"  saved {os.path.join(R, 'assistant_ppl_diag.json')}")
