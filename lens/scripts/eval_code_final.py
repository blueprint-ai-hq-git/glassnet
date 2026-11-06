"""eval_code_final.py — the corrected, apples-to-apples code evaluation.

WHAT WAS WRONG WITH MY LAST SWEEP (two bugs, both mine)
  1. It swept EVERY epoch checkpoint, so it ranked the best moment of each run instead of the final
     model — an unfair, cherry-picked comparison across runs.
  2. It used STYLE A prompts (cut at "def "), where the model must INVENT a function name. Names are
     the highest-entropy part of code, so the model emits <unk> and the snippet is dead before the
     body starts. reexec_code.py already documented this: "0/25 measured my prompt, not the model."
     The 8/25 baseline was measured with STYLE B (name and args GIVEN, model writes the body) --
     which is what code completion actually is.

THE TWO TASKS, BOTH REPORTED (neither hidden)
  STYLE A ("invent the signature")  : harder, arguably unfair to a 600K-param model
  STYLE B ("complete the body")     : what code completion means; the 8/25 baseline

Decoder settings matched to reexec_code.py so the numbers are comparable: n_new 200, temp 0.5,
top_k 15, rep 1.2.

GATES (pre-registered)
  k1  the best FINAL checkpoint beats 8/25 on STYLE B (the baseline's own task)
  k2  its ran-count is at least half its syntax-count
  k3  whichever way it goes, STYLE A and STYLE B are both printed -- a win on one is not a win
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import code_data                                          # noqa: E402
import lens                                               # noqa: E402
from lens import backend as B                             # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "code")
TOK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]")
N_EXEC = 25
N_NEW, TEMP, TOP_K, REP = 200, 0.5, 15, 1.2


def run_snippet(src, timeout=5):
    if len(src) > 20000:
        return None, "too long"
    try:
        ast.parse(src)
    except SyntaxError:
        return None, "syntax"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.py")
        open(p, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=td, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except Exception:
            return None, "err"
        return (r.returncode == 0), ("ok" if r.returncode == 0 else "runtime")


def gen(net, vocab, prompt, window, seed, n_new=N_NEW):
    vmap = {w: i for i, w in enumerate(vocab)}
    e = [vmap.get(t, 0) for t in TOK.findall(prompt.lower())]
    ctx = (e if len(e) >= window else [0] * (window - len(e)) + e)[-window:]
    out, used, r = [], {}, np.random.default_rng(seed)
    for _ in range(n_new):
        z = B.to_cpu(net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"])[0]
        z = z.astype(np.float64)
        for t, c in used.items():
            z[t] -= np.log(REP) * c
        zz = z / TEMP
        if TOP_K:
            cut = np.partition(zz, -TOP_K)[-TOP_K]
            zz = np.where(zz >= cut, zz, -np.inf)
        zz = zz - np.nanmax(zz[np.isfinite(zz)])
        p = np.exp(zz - zz.max())
        p = np.where(np.isfinite(p), p, 0.0)
        s = p.sum()
        p = p / s if s > 0 else np.ones(len(p)) / len(p)
        nxt = int(r.choice(len(p), p=p))
        tok = vocab[nxt] if nxt < len(vocab) else ""
        out.append(" " if tok == "<unk>" else tok)
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-window:]
    return " ".join(out)                                  # whitespace-preserving join


def two_style_prompts(vocab, ids, bounds, split, n=N_EXEC):
    """STYLE A: cut at 'def '. STYLE B: cut after the signature colon (body completion)."""
    a_list, b_list = [], []
    for j in range(len(bounds) - 1):
        if len(a_list) >= n:
            break
        if split[j] != 1:
            continue
        text = " ".join(vocab[i] for i in ids[bounds[j]:bounds[j + 1]])
        k = text.find("def ")
        if k < 0:
            continue
        a_list.append(text[:k + 4])
        rest = text[k:]
        colon = rest.find(":")
        b_list.append(rest[:colon + 1] if colon >= 0 else rest[:80])
    return a_list[:n], b_list[:n]


print("=" * 100)
print("CODE EVALUATION — FINAL checkpoints only, BOTH prompt styles, decoder matched to baseline")
print("=" * 100)

d = code_data.build()
ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
pa, pb = two_style_prompts(vocab, ids, bounds, split)
print(f"  {len(pa)} prompts each style · vocab {len(vocab):,} · decoder temp {TEMP} top_k {TOP_K}")

# FINAL checkpoint of each distinct run only (strip _epN, keep the highest N)
runs = {}
for f in os.listdir(R):
    m = re.match(r"^(.*)_ep(\d+)\.npz$", f)
    if m:
        tag, ep = m.group(1), int(m.group(2))
        if tag not in runs or ep > runs[tag][0]:
            runs[tag] = (ep, f)

rows = []
for tag, (ep, fn) in sorted(runs.items()):
    try:
        net = lens.load(os.path.join(R, fn), device="cpu")
    except Exception as e:
        print(f"  {tag}: load failed {type(e).__name__}")
        continue
    res = {}
    for style, prompts in (("A", pa), ("B", pb)):
        n_syn = n_run = 0
        ex = None
        for i, pr in enumerate(prompts):
            src = gen(net, vocab, pr, net.W, seed=i)
            ok, _ = run_snippet(src)
            if ok is not None:
                n_syn += 1
            if ok:
                n_run += 1
                if ex is None:
                    ex = src[:200]
        res[style] = {"syntax": n_syn, "ran": n_run, "total": len(prompts), "example": ex}
    rows.append({"tag": tag, "final_ep": ep, **{f"A_{k}": v for k, v in res["A"].items()},
                 **{f"B_{k}": v for k, v in res["B"].items()}})
    print(f"  {tag:<18} ep{ep:<3} A: {res['A']['syntax']:>2}/{len(pa)} syn "
          f"{res['A']['ran']:>2}/{len(pa)} ran | B: {res['B']['syntax']:>2}/{len(pb)} syn "
          f"{res['B']['ran']:>2}/{len(pb)} ran")

print("\n" + "-" * 100)
if not rows:
    print("  nothing loaded"); sys.exit(1)
bestB = max(rows, key=lambda r: (r["B_syntax"], r["B_ran"]))
bestA = max(rows, key=lambda r: (r["A_syntax"], r["A_ran"]))
print(f"  BEST on STYLE B (fair, = the 8/25 baseline task): {bestB['tag']} ep{bestB['final_ep']}"
      f"  syntax {bestB['B_syntax']}/{bestB['B_total']}  ran {bestB['B_ran']}/{bestB['B_total']}")
print(f"  BEST on STYLE A (invent the name):               {bestA['tag']} ep{bestA['final_ep']}"
      f"  syntax {bestA['A_syntax']}/{bestA['A_total']}  ran {bestA['A_ran']}/{bestA['A_total']}")
k1 = bestB["B_syntax"] > 8
k2 = bestB["B_ran"] >= max(1, bestB["B_syntax"] // 2)
print(f"\n  k1  style-B syntax {bestB['B_syntax']} > 8 (prior baseline) -> {'PASS' if k1 else 'FAIL'}")
print(f"  k2  style-B ran {bestB['B_ran']} >= half of parsed "
      f"({max(1, bestB['B_syntax']//2)}) -> {'PASS' if k2 else 'FAIL'}")
verdict = (f"{bestB['tag']} is the best code model on the FAIR task "
           f"({bestB['B_syntax']}/{bestB['B_total']} parse, {bestB['B_ran']}/{bestB['B_total']} run)."
           if k1 else
           "NO final checkpoint beats 8/25 on the fair task: coding stays weak. Reported, not spun.")
print(f"\n  VERDICT: {verdict}")
if bestB.get("B_example"):
    print(f"\n  winner example (style B):\n    {bestB['B_example']}")
json.dump(rows, open(os.path.join(R, "code_final_eval.json"), "w"), indent=1)
print(f"\n  saved {os.path.join(R, 'code_final_eval.json')}")
