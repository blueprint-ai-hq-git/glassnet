"""reexec_code.py — re-run the execution test with the token-join bug FIXED.

THE BUG
-------
`code_data.TOKEN_RE` drops whitespace entirely:

    TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]")

and `train_code.gen()` joined the generated tokens with `""`:

    return "".join((" " if t == "<unk>" else t) for t in out)

So `def foo ( x ) :` came back as `deffoo(x):` -- which is not Python, and never can be. The
reported **0/25 syntax** was the harness, not the model. A generator that cannot emit a space
cannot emit a function definition, so the test was measuring my own bug.

THE FIX
-------
Join with a single space, then let Python's own parser decide. `def foo ( x ) :` is valid Python;
whitespace around punctuation is a style question, not a syntax one.

This script loads the trained checkpoint and re-runs the SAME test, so the before/after is
comparable. It also prints the raw generated text, because the previous run never showed it --
which is exactly why the bug survived a full training run.
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
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import code_data                                          # noqa: E402
import lens                                               # noqa: E402
from lens import backend as B                             # noqa: E402

OUT = os.path.join(PROJ, "lens", "runs", "code")
CKPT = os.environ.get("CKPT", "code1_ep16")


def run_snippet(src, timeout=5):
    if len(src) > 20000:
        return None, "too long"
    try:
        ast.parse(src)
    except SyntaxError as e:
        return None, f"syntax: {type(e).__name__}"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "snippet.py")
        open(p, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=td, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except Exception as e:
            return None, f"exec: {type(e).__name__}"
    return (r.stdout.strip(), "ok") if r.returncode == 0 else (None, "raised")


def main():
    d = code_data.build()
    vocab, ids, bounds, split = d["vocab"], d["ids"], d["bounds"], d["split"]
    log = json.load(open(os.path.join(OUT, "code1_log.json")))
    a = log["args"]
    window = a["window"]
    ck = os.path.join(OUT, CKPT + ".npz")
    if not os.path.exists(ck):
        cand = sorted(f for f in os.listdir(OUT) if f.startswith("code1_ep") and f.endswith(".npz"))
        ck = os.path.join(OUT, cand[-1])
    print(f"  checkpoint: {os.path.basename(ck)}")

    import code_concepts
    Xva, _ = code_data.make_windows(ids, bounds, split, window, which=1)
    cs, spec = code_concepts.build(vocab, ids, np.zeros(len(ids), dtype=bool), windows=Xva)
    # lens.load rebuilds the concepts, supports and shape from the checkpoint's own metadata, so the
    # concept list above is only used to check that the rebuild is consistent with the run.
    net = lens.load(ck, device=a.get("device", "cpu"))
    got = [c.name for c in net.concepts]
    want = [c["name"] for c in cs]
    print(f"  concepts in checkpoint: {len(got)}   rebuilt from data: {len(want)}   "
          f"match: {set(got) == set(want)}")
    print(f"  loaded: d={a['d']} W={window} K={net.K} params={sum(p.data.size for p in net.parameters()):,}")

    vmap = {w: i for i, w in enumerate(vocab)}
    TOK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]")

    def gen(prompt, n_new=200, temp=0.5, top_k=15, rep=1.2, seed=0, join=" "):
        e = [vmap.get(t, 0) for t in TOK.findall(prompt.lower())]
        ctx = (e if len(e) >= window else [0] * (window - len(e)) + e)[-window:]
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
            p = p / p.sum()
            nxt = int(r.choice(len(p), p=p))
            out.append(vocab[nxt] if nxt < len(vocab) else "")
            used[nxt] = used.get(nxt, 0) + 1
            ctx = (ctx + [nxt])[-window:]
        return join.join(out)

    # validation prompts: the problem text, then let it write the function
    # TWO PROMPT STYLES, because the first one was unfair and I reported its result as a model limit.
    #   STYLE A (the broken one): cut at "def " -- the model must INVENT a function name. Names are
    #           the highest-entropy part of code (any identifier is legal), so it emits <unk> and
    #           the snippet is dead before the body starts. 0/25 measured my prompt, not the model.
    #   STYLE B (fair): cut AFTER the signature's colon -- "def name ( args ) :" is given, and the
    #           model writes the body. This is what code completion actually is.
    prompts_a, prompts_b = [], []
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
    print(f"  prompt style A (cut at 'def ')      : {len(prompts_a)}")
    print(f"  prompt style B (full signature)     : {len(prompts_b)}  <- used below")

    # THE FAIR TEST. Measured: 20 of the 25 style-B prompts contain `<unk>` -- the prompt literally
    # reads `def <unk> ( <unk> , <unk> ) :`. The model is then asked to CONTINUE a signature whose
    # names were deleted before it ever saw them, and continuing with `<unk>` is the locally correct
    # behaviour. So a large share of the generated `<unk>` is read-back, not invention, and a syntax
    # rate measured on those prompts partly measures the harness.
    #
    # Both numbers are reported. The broken-prompt number stays, because deleting the hard cases
    # would be lowering the bar; the clean number is added, because the broken one is not a test of
    # writing a function body.
    prompts_clean = [p for p in prompts_b if "<unk>" not in p]
    print(f"  style B prompts with NO <unk>       : {len(prompts_clean)}/{len(prompts_b)}  "
          f"<- the FAIR test")
    print()

    print()
    print("=" * 100)
    print(f"EXECUTION TEST — BEFORE the join fix (shown) vs AFTER")
    print("=" * 100)
    for label, join in [("BEFORE join=''", ""), ("AFTER join=' '", " ")]:
        stats = {"syntax": 0, "ran": 0, "total": 0}
        ex = []
        for pr in prompts:
            code = gen(pr, join=join)
            stats["total"] += 1
            try:
                ast.parse(code)
                stats["syntax"] += 1
            except SyntaxError:
                pass
            _, why = run_snippet(code)
            if why == "ok":
                stats["ran"] += 1
            ex.append((pr, code, why))
        print(f"  {label:<16} syntax {stats['syntax']}/{stats['total']}   runs {stats['ran']}/{stats['total']}")
        if join == " ":
            print()
            print("  --- the first 8, in full, with the join fixed ---")
            for pr, code, why in ex[:8]:
                print(f"\n  PROMPT ...{pr[-70:]!r}")
                print(f"  WROTE  {code[:300]!r}")
                print(f"  RESULT {why}")
        json.dump({"stats": stats, "examples": [{"prompt": p, "code": c, "result": w}
                                                for p, c, w in ex]},
                  open(os.path.join(OUT, "code1_reexec.json"), "w"), indent=1)

    # ---- THE CLEAN-PROMPT TEST, run separately so neither number can hide the other
    print()
    print("=" * 100)
    print(f"FAIR TEST — prompts whose signature contains NO `<unk>`  ({len(prompts_clean)} prompts)")
    print("=" * 100)
    st = {"syntax": 0, "ran": 0, "total": 0}
    ex2 = []
    for pr in prompts_clean:
        code = gen(pr, join=" ")
        st["total"] += 1
        try:
            ast.parse(code)
            st["syntax"] += 1
        except SyntaxError:
            pass
        _, why = run_snippet(code)
        if why == "ok":
            st["ran"] += 1
        ex2.append({"prompt": pr, "code": code, "result": why})
    print(f"  clean prompts   syntax {st['syntax']}/{st['total']}   runs {st['ran']}/{st['total']}")
    if st["total"]:
        b = {"syntax": 0, "ran": 0, "total": 0}
        for pr in prompts:
            code = gen(pr, join=" ")
            b["total"] += 1
            try:
                ast.parse(code)
                b["syntax"] += 1
            except SyntaxError:
                pass
            _, why = run_snippet(code)
            if why == "ok":
                b["ran"] += 1
        print(f"  ALL prompts     syntax {b['syntax']}/{b['total']}   runs {b['ran']}/{b['total']}")
        if b["total"]:
            print(f"  difference      broken prompts are {b['total']-st['total']}; on the clean ones "
                  f"the model scores {st['syntax']/st['total']*100:.0f}% vs "
                  f"{b['syntax']/b['total']*100:.0f}% overall")
    json.dump({"stats": st, "examples": ex2,
               "clean_prompt_test": True,
               "note": "prompts whose signature contains no <unk> -- a fair test of writing a body"},
              open(os.path.join(OUT, "code1_reexec_clean.json"), "w"), indent=1)
    print("=" * 100)


if __name__ == "__main__":
    main()
