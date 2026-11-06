"""code_scale_test.py — the SAME comparison at 150 prompts instead of 25, to settle whether the
32k vocabulary really helps.

WHY: the 25-prompt test gave 4/25 vs 6/25 and Fisher exact said p=0.725 — unreadable. With 768
validation documents available, n=150 makes an 8-point difference detectable. The measurement was
the bottleneck, so fix the measurement.

DESIGN (pre-registered, fixed BEFORE running)
  * n = 150 prompts per model, built by the SAME construction for both (style B, cut after the
    signature colon, from the full document text) — the fair body-completion task
  * each checkpoint evaluated with its OWN vocabulary (the bug that invalidated the last run)
  * same decoder settings for both (n_new 200, temp 0.5, top_k 15, rep 1.2)
  * syntax by CPython ast.parse, execution by real subprocess
  * Fisher exact test on the 2x2 table, reported WITH the raw counts

GATES
  r1  the syntax difference is significant at p < 0.05  -> vocabulary genuinely helps
  r2  if p >= 0.05 -> NO significant difference; report that vocabulary is NOT the blocker and stop
      pursuing vocabulary changes. Both outcomes are informative; neither gets spun.
  r3  <unk> rate reported for both, since that is the mechanism the vocabulary change should move
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from math import comb

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens                                               # noqa: E402
from lens import backend as B                             # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "code")
CI = os.path.join(PROJ, "data", "code", "prepared")
TOK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]")
N = int(os.environ.get("N_PROMPTS", "150"))
MODELS = [("code1_ep16", "v6000_both"), ("covocab32k_ep16", "v32000_code")]


def run_snippet(src, timeout=5):
    if len(src) > 20000:
        return None
    try:
        ast.parse(src)
    except SyntaxError:
        return None
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.py")
        open(p, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=td, capture_output=True,
                               text=True, timeout=timeout)
        except Exception:
            return None
        return r.returncode == 0


def gen(net, vocab, prompt, seed, n_new=200, temp=0.5, top_k=15, rep=1.2):
    vmap = {w: i for i, w in enumerate(vocab)}
    e = [vmap.get(t, 0) for t in TOK.findall(prompt.lower())]
    W = net.W
    ctx = (e if len(e) >= W else [0] * (W - len(e)) + e)[-W:]
    out, used, r = [], {}, np.random.default_rng(seed)
    for _ in range(n_new):
        z = B.to_cpu(net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"])[0]
        z = z.astype(np.float64)
        for t, c in used.items():
            z[t] -= np.log(rep) * c
        zz = z / temp
        if top_k:
            cut = np.partition(zz, -top_k)[-top_k]
            zz = np.where(zz >= cut, zz, -np.inf)
        zz = zz - np.nanmax(zz[np.isfinite(zz)])
        p = np.exp(zz - zz.max())
        p = np.where(np.isfinite(p), p, 0.0)
        s = p.sum()
        p = p / s if s > 0 else np.ones(len(p)) / len(p)
        nxt = int(r.choice(len(p), p=p))
        tok = vocab[nxt] if nxt < len(vocab) else ""
        out.append(tok)
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-W:]
    return out, " ".join(" " if t == "<unk>" else t for t in out)


def prompts_for(vocab, data_tag, n=N):
    ids = np.load(os.path.join(CI, f"code_ids_{data_tag}.npy"))
    bounds = np.load(os.path.join(CI, f"code_bounds_{data_tag}.npy"))
    split = np.load(os.path.join(CI, f"code_split_{data_tag}.npy"))
    out = []
    for j in range(len(bounds) - 1):
        if len(out) >= n:
            break
        if split[j] != 1:
            continue
        text = " ".join(vocab[i] for i in ids[bounds[j]:bounds[j + 1]])
        k = text.find("def ")
        if k < 0:
            continue
        colon = text.find(" : ", k)
        if colon > 0:
            out.append(text[:colon + 3])
    return out


def fisher_two_sided(a, b, c, d):
    n = a + b + c + d
    if n == 0:
        return 1.0

    def p_of(x):
        try:
            return comb(a + c, x) * comb(b + d, a + b - x) / comb(n, a + b)
        except Exception:
            return 0.0
    obs = p_of(a)
    tot = 0.0
    for x in range(0, min(a + c, a + b) + 1):
        px = p_of(x)
        if px <= obs + 1e-12:
            tot += px
    return min(1.0, tot)


print("=" * 100)
print(f"CODE SCALE TEST — n={N} prompts per model (was 25), fair style-B task")
print("=" * 100)

results = {}
for tag, dtag in MODELS:
    f = os.path.join(R, tag + ".npz")
    if not os.path.exists(f):
        print(f"  {tag}: missing"); continue
    net = lens.load(f, device="cpu")
    vf = [p for p in os.listdir(CI) if p.startswith("code_vocab")]
    vocab = None
    for c in vf:
        v = json.load(open(os.path.join(CI, c)))
        if len(v) == net.V:
            vocab = v
            break
    if vocab is None:
        print(f"  {tag}: no vocab of width {net.V} — refusing")
        continue
    prompts = prompts_for(vocab, dtag)
    syn = run = unk_tok = tot_tok = 0
    for i, pr in enumerate(prompts):
        toks, src = gen(net, vocab, pr, seed=i)
        unk_tok += sum(1 for t in toks if t == "<unk>")
        tot_tok += len(toks)
        ok = run_snippet(src)
        if ok is not None:
            syn += 1
        if ok:
            run += 1
    results[tag] = {"syntax": syn, "ran": run, "total": len(prompts),
                    "unk": unk_tok / max(tot_tok, 1), "V": net.V}
    print(f"  {tag:<18} V={net.V:>6,}  syntax {syn:>3}/{len(prompts)}  ran {run:>3}/{len(prompts)}"
          f"  unk {unk_tok/max(tot_tok,1)*100:5.1f}%")

a = results.get("code1_ep16")
b = results.get("covocab32k_ep16")
print("\n" + "-" * 100)
if a and b:
    p = fisher_two_sided(b["syntax"], b["total"] - b["syntax"],
                         a["syntax"], a["total"] - a["syntax"])
    print(f"  code1   (6k) : {a['syntax']}/{a['total']} = {a['syntax']/a['total']*100:.1f}%  "
          f"unk {a['unk']*100:.1f}%")
    print(f"  32k vocab    : {b['syntax']}/{b['total']} = {b['syntax']/b['total']*100:.1f}%  "
          f"unk {b['unk']*100:.1f}%")
    print(f"  Fisher exact two-sided p = {p:.4f}")
    r1 = p < 0.05
    print(f"\n  r1  difference significant at p<0.05 -> {'PASS' if r1 else 'FAIL'}")
    if r1:
        verdict = (f"VOCABULARY GENUINELY HELPS: {a['syntax']}/{a['total']} -> "
                   f"{b['syntax']}/{b['total']}, p={p:.4f}. Real, measurable improvement.")
    else:
        verdict = (f"NO SIGNIFICANT DIFFERENCE (p={p:.4f}): the vocabulary change does NOT "
                   f"demonstrably improve code generation. Stop pursuing vocabulary as the fix; "
                   f"the remaining cause is model capacity or training data content.")
    print(f"\n  VERDICT: {verdict}")
    print(f"  r3  unk {a['unk']*100:.1f}% -> {b['unk']*100:.1f}% (the mechanism the change targets)")
    json.dump({"code1": a, "covocab32k": b, "fisher_p": p, "gates": {"r1": bool(r1)},
               "verdict": verdict}, open(os.path.join(R, "code_scale_test.json"), "w"), indent=1)
    print(f"  saved {os.path.join(R, 'code_scale_test.json')}")
else:
    print("  both models required — one is missing")
