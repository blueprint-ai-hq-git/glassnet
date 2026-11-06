"""compare_unk_rate.py — the decisive test of change 3, done fairly.

TWO PROBLEMS WITH WHAT WAS REPORTED
1. "clean prompts 2/5 vs 1/3" compared DIFFERENT prompt sets. The earlier fair test ran at 32k vocab
   (more tokens known -> 5 clean prompts); the drop-unk run is at 6k (fewer known -> 3 clean prompts).
   Different denominators, so the percentages are not comparable. The baseline must be re-measured at
   the SAME vocabulary.

2. The prediction being tested was "the generated `<unk>` rate falls sharply", and the `<unk>` rate of
   the GENERATED TEXT was never measured -- only syntax rates were. Syntax is an indirect proxy.
   Measure the thing that was predicted.

So: load both 6k checkpoints, generate from the SAME prompts, and count `<unk>` in the output.
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

os.environ["CODE_DOC"] = "code"
os.environ["CD_VOCAB"] = "6000"

import code_data                                          # noqa: E402
import lens                                               # noqa: E402
from lens import backend as B                             # noqa: E402
import lens.tensor as T                                   # noqa: E402

OUT = os.path.join(PROJ, "lens", "runs", "code")
W = 16

print("=" * 100)
print("CHANGE 3, MEASURED PROPERLY: both checkpoints at 6k vocab, same prompts")
print("=" * 100)

d = code_data.build()
vocab, ids, bounds, split = d["vocab"], d["ids"], d["bounds"], d["split"]
vmap = {w: i for i, w in enumerate(vocab)}
TOK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]")

# the SAME validation prompts both models see
prompts_b, prompts_a = [], []
for j in range(len(bounds) - 1):
    if split[j] != 1:
        continue
    text = " ".join(vocab[i] for i in ids[bounds[j]:bounds[j + 1]])
    k = text.find("def ")
    if k < 0:
        continue
    if len(prompts_a) < 25:
        prompts_a.append(text[:k + 4])
    colon = text.find(" : ", k)
    if colon > 0 and len(prompts_b) < 25:
        prompts_b.append(text[:colon + 3])
prompts = prompts_b
clean = [p for p in prompts if "<unk>" not in p]
print(f"  {len(prompts)} prompts, of which {len(clean)} contain no <unk> "
      f"(both models tested on the SAME sets)")


def gen(net, prompt, n_new=200, temp=0.5, top_k=15, rep=1.2, seed=0):
    """Returns (text, token_ids).

    WHY IDS ARE RETURNED TOO: the first version of this measurement counted `<unk>` by re-tokenising
    the generated text with TOK. But TOK splits `<unk>` into ['<', 'unk', '>'], so the pattern can
    never match the literal token and the count came back as a clean 0.00% for every model -- a
    silent zero that looked like a perfect result. The visible text was full of `<unk>`, which is
    what exposed it. Count the IDs the model actually chose; never re-tokenise to measure a token.
    """
    e = [vmap.get(t, 0) for t in TOK.findall(prompt.lower())]
    ctx = (e if len(e) >= W else [0] * (W - len(e)) + e)[-W:]
    out, ids_out, used = [], [], {}
    r = np.random.default_rng(seed)
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
        p = p / p.sum()
        nxt = int(r.choice(len(p), p=p))
        out.append(vocab[nxt] if nxt < len(vocab) else "")
        ids_out.append(nxt)
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-W:]
    return " ".join(out), ids_out


def run_snip(src, timeout=5):
    if len(src) > 20000:
        return "too long"
    try:
        ast.parse(src)
    except SyntaxError:
        return "syntax"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.py")
        open(p, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=td, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout"
    return "ok" if r.returncode == 0 else "raised"


MODELS = [("codedoc_full  (no mask)", "codedoc_full_ep16.npz"),
          ("codropunk     (masked)", "codropunk_ep16.npz")]

print()
print(f"  {'model':<26}{'<unk>/gen tok':>14}{'syntax':>9}{'runs':>8}{'clean syntax':>14}{'clean runs':>12}")
results = {}
for tag, ck in MODELS:
    path = os.path.join(OUT, ck)
    if not os.path.exists(path):
        print(f"  {tag:<26}  checkpoint missing")
        continue
    net = lens.load(path, device="cpu")
    tot_unk = tot_tok = 0
    syn = ran = 0
    csyn = cran = 0
    for pr in prompts:
        code, gids = gen(net, pr)
        tot_tok += len(gids)
        tot_unk += sum(1 for i in gids if i == 0)      # id 0 IS <unk>; do not re-tokenise
        why = run_snip(code)
        if why != "syntax":
            syn += 1
        if why == "ok":
            ran += 1
    for pr in clean:
        code, _ = gen(net, pr)
        why = run_snip(code)
        if why != "syntax":
            csyn += 1
        if why == "ok":
            cran += 1
    results[tag] = {"unk_rate": tot_unk / max(tot_tok, 1), "syn": syn, "ran": ran,
                    "csyn": csyn, "cran": cran, "n": len(prompts), "nc": len(clean)}
    print(f"  {tag:<26}{tot_unk/max(tot_tok,1)*100:>13.2f}%{syn:>6}/{len(prompts):<3}{ran:>5}/{len(prompts):<3}"
          f"{csyn:>11}/{len(clean):<3}{cran:>9}/{len(clean):<3}")

print()
keys = list(results)
if len(keys) == 2:
    a, b = results[keys[0]], results[keys[1]]
    print("  THE COMPARISON (identical prompts, identical vocab, identical budget):")
    print(f"    generated <unk> rate   {a['unk_rate']*100:6.2f}%  ->  {b['unk_rate']*100:6.2f}%   "
          f"({(b['unk_rate']-a['unk_rate'])*100:+.2f} points)")
    print(f"    syntax (all prompts)   {a['syn']:>3}/{a['n']}   ->  {b['syn']:>3}/{b['n']}")
    print(f"    runs   (all prompts)   {a['ran']:>3}/{a['n']}   ->  {b['ran']:>3}/{b['n']}")
    print(f"    syntax (clean prompts) {a['csyn']:>3}/{a['nc']}   ->  {b['csyn']:>3}/{b['nc']}")
    print(f"    runs   (clean prompts) {a['cran']:>3}/{a['nc']}   ->  {b['cran']:>3}/{b['nc']}")
    print()
    print("  PREDICTIONS CHECKED:")
    print(f"    1. generated <unk> falls sharply : {'HELD' if b['unk_rate'] < a['unk_rate']*0.75 else 'NOT HELD'}"
          f"  ({a['unk_rate']*100:.2f}% -> {b['unk_rate']*100:.2f}%)")
    print(f"    2. executables rise              : "
          f"{'HELD' if b['ran'] > a['ran'] else 'NOT HELD'}  ({a['ran']} -> {b['ran']} of {a['n']})")
    print(f"    3. ppl gets worse                : HELD  (39.05 -> 67.02, in the training log)")
json.dump(results, open(os.path.join(OUT, "unk_rate_compare.json"), "w"), indent=1)
print("=" * 100)
