"""eval_all_code.py — run EVERY code checkpoint through the SAME corrected harness, so the best one
is chosen on evidence instead of on perplexity (which has already lied twice here).

WHY THIS EXISTS
  ppl ranked codedoc_full (39.05) best; the exec test ranked it 0/25 at one point and 8/25 at another
  purely because the harness changed. ppl and runnability disagree, so runnability decides.

THE HARNESS (fixed, and identical for every checkpoint)
  * whitespace-preserving join  (the bug that produced a false 0/25 twice)
  * prompt = the validation text cut at the first 'def ' -- same construction train_code uses
  * syntax judged by CPython's ast.parse, execution by a real subprocess
  * SAME 25 prompts, SAME seeds, SAME decoder settings (temp 0.6, top-k 20, rep 1.25)

GATES (pre-registered)
  g1  at least one checkpoint must beat 8/25 syntax  (otherwise coding stays broken and I say so)
  g2  the winner must have ran >= syntax/2         (parsing without running is not "writing code")
  g3  the winner's ppl must be reported ALONGSIDE its exec rate, never instead of it
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
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]")
N_EXEC = 25
N_NEW = 220


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
            return None, "exec-error"
        return (r.returncode == 0), ("ok" if r.returncode == 0 else "runtime")


def gen(net, vocab, prompt, window, seed, n_new=N_NEW, temp=0.6, top_k=20, rep=1.25):
    vmap = {w: i for i, w in enumerate(vocab)}
    e = [vmap.get(t, 0) for t in TOKEN_RE.findall(prompt.lower())]
    ctx = (e if len(e) >= window else [0] * (window - len(e)) + e)[-window:]
    out, used, r = [], {}, np.random.default_rng(seed)
    for _ in range(n_new):
        lg = B.to_cpu(net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"])[0]
        z = lg.astype(np.float64)
        for t, cnt in used.items():
            z[t] -= np.log(rep) * cnt
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
        tok = vocab[nxt] if nxt < len(vocab) else " "
        out.append(" " if tok == "<unk>" else tok)
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-window:]
    return " ".join(out)                                  # whitespace-preserving join


def prompts_for(vocab, ids, bounds, split, n=N_EXEC):
    evals = []
    for j in range(len(bounds) - 1):
        if len(evals) >= n:
            break
        if split[j] != 1:
            continue
        text = " ".join(vocab[i] for i in ids[bounds[j]:bounds[j + 1]])
        if "def " not in text:
            continue
        evals.append(text[:text.find("def ") + 4])
    return evals


print("=" * 100)
print("ALL CODE CHECKPOINTS — one corrected harness, exec rate decides")
print("=" * 100)

d = code_data.build()
ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
prompts = prompts_for(vocab, ids, bounds, split)
print(f"  {len(prompts)} shared prompts · vocab {len(vocab):,}")

CKPTS = [f[:-4] for f in sorted(os.listdir(R)) if f.endswith(".npz") and f.startswith("co")]
CKPTS += [f[:-4] for f in sorted(os.listdir(R)) if f.endswith(".npz") and f.startswith("lad_code")]
CKPTS += [f[:-4] for f in sorted(os.listdir(R)) if f.endswith(".npz") and f.startswith("cfix_")]
CKPTS = sorted(set(CKPTS))
print(f"  checkpoints: {CKPTS}")

rows = []
for tag in CKPTS:
    path = os.path.join(R, tag + ".npz")
    try:
        net = lens.load(path, device="cpu")
    except Exception as e:
        print(f"  {tag}: LOAD FAILED {type(e).__name__}")
        continue
    n_syn = n_run = 0
    ex = None
    for i, pr in enumerate(prompts):
        src = gen(net, vocab, pr, net.W, seed=i)
        ok, why = run_snippet(src)
        if ok is not None:
            n_syn += 1
        if ok:
            n_run += 1
            if ex is None:
                ex = src[:180]
    ppl = None
    lf = os.path.join(R, tag + "_log.json")
    if os.path.exists(lf):
        try:
            ppl = json.load(open(lf))["history"][-1]["val_ppl"]
        except Exception:
            pass
    rows.append({"tag": tag, "syntax": n_syn, "ran": n_run, "total": len(prompts),
                 "ppl": ppl, "example": ex})
    print(f"  {tag:<18} syntax {n_syn:>2}/{len(prompts)}  ran {n_run:>2}/{len(prompts)}"
          f"  ppl {('%.2f' % ppl) if ppl else 'n/a'}")

print("\n" + "-" * 100)
if not rows:
    print("  NO CHECKPOINT LOADED — nothing to rank")
    sys.exit(1)
best_syn = max(rows, key=lambda r: (r["syntax"], r["ran"]))
print(f"  BEST BY SYNTAX: {best_syn['tag']}  syntax {best_syn['syntax']}/{best_syn['total']}  "
      f"ran {best_syn['ran']}/{best_syn['total']}  ppl {best_syn['ppl']}")
print(f"  BEST BY PPL:    "
      f"{max((r for r in rows if r['ppl']), key=lambda r: -r['ppl'])['tag'] if any(r['ppl'] for r in rows) else 'n/a'}")
g1 = best_syn["syntax"] > 8
g2 = best_syn["ran"] >= max(1, best_syn["syntax"] // 2)
print(f"\n  g1  best syntax {best_syn['syntax']} > 8 (prior baseline) -> {'PASS' if g1 else 'FAIL'}")
print(f"  g2  ran {best_syn['ran']} >= half of parsed ({max(1, best_syn['syntax']//2)}) "
      f"-> {'PASS' if g2 else 'FAIL'}")
if g1 and g2:
    verdict = f"{best_syn['tag']} is the best code model on EVIDENCE (exec rate, not ppl)."
else:
    verdict = ("NO checkpoint clears the bar: coding remains weak. The best available is "
               f"{best_syn['tag']} at {best_syn['syntax']}/{best_syn['total']}. Reported, not spun.")
print(f"\n  VERDICT: {verdict}")
if best_syn.get("example"):
    print(f"\n  winner's best example:\n    {best_syn['example']}")
json.dump(rows, open(os.path.join(R, "all_code_eval.json"), "w"), indent=1)
print(f"\n  saved {os.path.join(R, 'all_code_eval.json')}")
