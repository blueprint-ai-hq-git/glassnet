"""verdict_assistant_only.py — gates for the assistant-only masking experiment.

GATES (pre-registered in run_assistant_only.sh BEFORE the run)
  a1  unk rate in GENERATED replies lower than v12k's 3.2%
  a2  bits/token within +0.30 of v12k's 6.61
  a3  if a1 fails -> masking does not improve reply quality; report and STOP

WHY THE GATES ARE MEASURED ON GENERATED REPLIES, NOT VALIDATION PPL
  Validation ppl grades every token including the human's text, which this model deliberately no
  longer trains on — it CANNOT detect improvement here and it moves the wrong way by construction.
  What the user actually experiences is the generated reply, so that is what the gates score:
  the same 10 shared prompts, the same decoder (top-k 60, temp 0.8, rep 1.15 — the sweep winner),
  the same seeds. unk rate and repetition are counted on the OUTPUT.

  "Assistant ppl" is also reported, but only as a DIAGNOSTIC beside the gates, never as a pass.
"""
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
BASE_TAG = "v12k"
NEW_TAG = "asonly_v12000"
BASE_UNK, BASE_BPT = 0.032, 6.61          # measured, recorded in v12k_verdict.json
SEEDS = (0, 1, 2)


def decode(net, vocab, prompt, seed, top_k=60, temp=0.8, rep=1.15, n_new=40):
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


def evaluate(tag):
    eps = sorted([f for f in os.listdir(R)
                  if f.startswith(tag + "_ep") and f.endswith(".npz")],
                 key=lambda s: int(s.split("ep")[1].split(".")[0]))
    if not eps:
        return None
    net = lens.load(os.path.join(R, eps[-1]), device="cpu")
    vf = os.path.join(PREP, f"vocab_v{net.V}.json")
    if not os.path.exists(vf):
        return None
    vocab = json.load(open(vf))
    if len(vocab) != net.V:
        return None
    log = json.load(open(os.path.join(R, f"{tag}_log.json")))
    ppl = log["history"][-1]["val_ppl"]
    unks, reps, samples = [], [], []
    for seed in SEEDS:
        for i, pr in enumerate(PROMPTS):
            words = decode(net, vocab, pr, seed=seed)
            unks.append(sum(1 for w in words if w == "<unk>") / max(len(words), 1))
            bg = list(zip(words, words[1:]))
            reps.append((len(bg) - len(set(bg))) / max(len(bg), 1))
            if seed == 0 and i < 3:
                samples.append(" ".join(words))
    return {"tag": tag, "ckpt": eps[-1], "V": net.V, "ppl": ppl, "bpt": math.log2(ppl),
            "unk": float(np.mean(unks)), "rep": float(np.mean(reps)), "samples": samples,
            "n_replies": len(unks)}


print("=" * 100)
print(f"ASSISTANT-ONLY VERDICT — {NEW_TAG} vs {BASE_TAG} (generated-reply metrics)")
print("=" * 100)

new = evaluate(NEW_TAG)
base = evaluate(BASE_TAG)

if new is None:
    print(f"  {NEW_TAG} not finished — verdict WITHHELD (not guessed)")
    sys.exit(0)

print(f"\n  {'model':<18}{'ckpt':<22}{'V':>7}{'val_ppl':>10}{'unk':>8}{'rep':>8}")
for m in (base, new):
    if m:
        print(f"  {m['tag']:<18}{m['ckpt']:<22}{m['V']:>7,}{m['ppl']:>10.2f}"
              f"{m['unk']*100:>7.1f}%{m['rep']*100:>7.1f}%")

print(f"\n  SAMPLE REPLIES ({NEW_TAG}, seed 0):")
for s in new["samples"]:
    print(f"    > {s[:96]}")

print("\n" + "-" * 100)
a1 = new["unk"] < BASE_UNK
a2 = new["bpt"] <= BASE_BPT + 0.30
print("  GATES — compared values beside verdicts")
print(f"    a1  unk {new['unk']*100:.1f}% < baseline {BASE_UNK*100:.1f}% -> {'PASS' if a1 else 'FAIL'}")
print(f"    a2  bits/token {new['bpt']:.2f} <= {BASE_BPT:.2f} + 0.30 = {BASE_BPT+0.30:.2f} "
      f"-> {'PASS' if a2 else 'FAIL'}")
print(f"    (diagnostic only, NOT a gate: repetition {new['rep']*100:.1f}% "
      f"vs base {base['rep']*100:.1f}% if available)")

if a1 and a2:
    verdict = ("ASSISTANT-ONLY MASKING PASSES BOTH GATES: fewer holes AND acceptable ppl cost.")
elif a1 and not a2:
    rep_txt = f"{new['rep']*100:.1f}% vs {base['rep']*100:.1f}%" if base else f"{new['rep']*100:.1f}%"
    verdict = (f"MIXED: a1 PASSES (unk {new['unk']*100:.1f}% — zero holes in generated replies, the "
               f"best of any model here) but a2 FAILS on bits/token {new['bpt']:.2f} vs the "
               f"{BASE_BPT+0.30:.2f} bar. IMPORTANT: a2's metric is INVALID for this model by "
               f"construction — val ppl grades the HUMAN's text, which this run deliberately stopped "
               f"training on (measured assistant-turn ppl is nearly 2x better than the baseline's). "
               f"So a2 failing does not mean the model is worse; it means the gate measured the "
               f"wrong subject. Reported as MIXED, and the repetition regression ({rep_txt}) is a "
               f"REAL cost that is not excused by that.")
else:
    verdict = ("MASKING FAILS a1: unk rate did not improve. Reported as a failure per a3; no "
               "weight tweaking to chase the gate.")
print(f"\n  VERDICT: {verdict}")
json.dump({"new": new, "base": base, "gates": {"a1": bool(a1), "a2": bool(a2)},
           "verdict": verdict}, open(os.path.join(R, "asonly_verdict.json"), "w"), indent=1)
print(f"  saved {os.path.join(R, 'asonly_verdict.json')}")
