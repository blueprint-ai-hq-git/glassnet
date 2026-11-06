"""test_constrained_code.py — constrained decoding ON THE REAL CODE MODEL.

THE QUESTION (pre-registered, binary, fixed before running):
  Does masking syntactically-invalid tokens raise the syntax rate above the 8/25 baseline?

GATES
  c1  syntax rate with constraint > 8/25 (the measured unconstrained baseline, same 25 prompts,
      same checkpoint, same decoder settings)
  c2  of the constrained outputs that parse, at least half must also RUN (constrained != degenerate)
  c3  c1 passes but c2 fails -> the constraint produces valid-but-dead code; report honestly and
      keep the unconstrained model as the reference rather than pretending the constraint won.

INTERFACE (read from code_data.py and train_code.py, not guessed):
  code_data.build() -> {"ids", "vocab", "bounds", "split", ...}
  tokenize via re.findall(TOKEN_RE, text.lower())   -- the same regex train_code.enc uses
  code1_ep16.npz is the checkpoint the 8/25 baseline came from

Everything is measured against CPython's own ast.parse and real subprocess execution — the same
harness that produced the 8/25 baseline, so the two numbers are comparable.
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
from constrain_code import SyntaxConstraint, mask_invalid  # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "code")
CKPT = os.environ.get("CKPT", "code1_ep16")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]")
N_EXEC = 25


def build_prompts(vocab, ids, bounds, split, n=N_EXEC):
    """The same construction train_code.main() uses: validation pairs, text cut at 'def '."""
    evals = []
    for j in range(len(bounds) - 1):
        if len(evals) >= n:
            break
        if split[j] != 1:
            continue
        seg = ids[bounds[j]:bounds[j + 1]]
        text = " ".join(vocab[i] for i in seg)
        if "def " not in text:
            continue
        evals.append(text[:text.find("def ") + 4])
    return evals


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
        return (r.returncode == 0), (r.stderr.strip().splitlines()[-1][:60] if r.returncode else "ok")


def gen(net, vocab, prompt, window, n_new=220, temp=0.6, top_k=20, rep=1.25, seed=0,
        constrain=True):
    vmap = {w: i for i, w in enumerate(vocab)}
    e = [vmap.get(t, 0) for t in TOKEN_RE.findall(prompt.lower())]
    ctx = (e if len(e) >= window else [0] * (window - len(e)) + e)[-window:]
    c = SyntaxConstraint() if constrain else None
    out, used, r = [], {}, np.random.default_rng(seed)
    for _ in range(n_new):
        lg = B.to_cpu(net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"])[0]
        z = lg.astype(np.float64)
        for t, cnt in used.items():
            z[t] -= np.log(rep) * cnt
        if constrain:
            z = mask_invalid(z, vocab, c)
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
        if constrain:
            c.feed(tok)
        out.append(" " if tok == "<unk>" else tok)
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-window:]
    # WHITESPACE-PRESERVING JOIN. `TOKEN_RE` drops whitespace, so tokens arrive as
    # ['def','__init__','(', ...] and joining with "" yields "def__init__(" which can never parse.
    # This is the exact bug that made an earlier run report 0/25. Join with a space; CPython's parser
    # decides whether the result is valid (`def foo ( x ) :` is valid Python).
    return " ".join(out)


print("=" * 100)
print(f"CONSTRAINED DECODING — {CKPT}")
print("=" * 100)

d = code_data.build()
ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
net = lens.load(os.path.join(R, CKPT + ".npz"), device="cpu")
window = net.W
prompts = build_prompts(vocab, ids, bounds, split)
print(f"  {len(prompts)} prompts · window {window} · vocab {len(vocab):,}")

results = {}
for mode, constrain in (("unconstrained", False), ("constrained", True)):
    n_syn = n_run = 0
    samples = []
    for i, pr in enumerate(prompts):
        src = gen(net, vocab, pr, window, seed=i, constrain=constrain)
        ok, why = run_snippet(src)
        if ok is not None:
            n_syn += 1
        if ok:
            n_run += 1
        if i < 2:
            samples.append({"prompt": pr[:44], "code": src[:200], "syntax": ok is not None,
                            "ran": ok})
    results[mode] = {"syntax": n_syn, "ran": n_run, "total": len(prompts), "samples": samples}
    print(f"\n  {mode:<14} syntax {n_syn}/{len(prompts)}   ran {n_run}/{len(prompts)}")
    for s in samples:
        print(f"    syntax={s['syntax']} ran={s['ran']}")
        print(f"    {s['code'][:140]}")

u, c = results["unconstrained"], results["constrained"]
c1 = c["syntax"] > u["syntax"]
c2 = (c["ran"] >= max(1, c["syntax"] // 2)) if c["syntax"] else False
print("\n" + "-" * 100)
print("  GATES — compared values beside verdicts")
print(f"    c1  syntax {c['syntax']}/{c['total']} > unconstrained {u['syntax']}/{u['total']} "
      f"-> {'PASS' if c1 else 'FAIL'}")
print(f"    c2  ran {c['ran']}/{c['total']} >= half of parsed ({max(1, c['syntax']//2)}) "
      f"-> {'PASS' if c2 else 'FAIL'}")
if c1 and c2:
    verdict = "CONSTRAINED DECODING WORKS: syntax up and the code still runs."
elif c1:
    verdict = "MIXED: syntax rose but the code does not run -> valid-but-dead; report honestly."
else:
    verdict = ("CONSTRAINED DECODING FAILS: no syntax improvement -> report as failure, keep the "
               "unconstrained model as reference.")
print(f"\n  VERDICT: {verdict}")
json.dump(results, open(os.path.join(R, "constrained_replies.json"), "w"), indent=1)
print(f"  saved {os.path.join(R, 'constrained_replies.json')}")
