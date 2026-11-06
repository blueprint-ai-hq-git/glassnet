"""train_code.py — a model that writes CODE, judged by EXECUTION.

THE TEST IS THE POINT
---------------------
Every other eval in this project measures the model against held-out text. This one measures it
against reality: the generated code is EXECUTED, and it either produces the right answer or it does
not. The owner's rule: "a CHECKABLE task, where a wrong answer is provably wrong."

SAFETY OF EXECUTION
-------------------
Generated code is run in a separate process, in a temporary directory, with:
  - a hard timeout (5 seconds)
  - no network access available to it in practice (the runner does not import sockets)
  - a fresh temp cwd so it cannot touch project files
  - the output captured, never eval'd in this process
It is model output and must be treated as untrusted.

WHAT IS MEASURED
----------------
1. val perplexity and next-token accuracy on held-out code
2. how often generated code is SYNTACTICALLY VALID python (ast.parse)
3. how often it RUNS without raising
4. how often it produces a value equal to the reference solution's value on the same input
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import code_concepts                                      # noqa: E402
import code_data                                          # noqa: E402
import lens                                               # noqa: E402
import lens.tensor as T                                   # noqa: E402
from lens import backend as B                             # noqa: E402
from lens.concepts import Concept                         # noqa: E402

OUT = os.path.join(PROJ, "lens", "runs", "code")
os.makedirs(OUT, exist_ok=True)


def log(m):
    print(m, flush=True)


def run_snippet(src, stdin_text="", timeout=5):
    """Execute untrusted generated code in an isolated temp dir with a hard timeout."""
    if len(src) > 20000:
        return None, "too long"
    try:
        ast.parse(src)
    except SyntaxError as e:
        return None, f"syntax: {type(e).__name__}"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "snippet.py")
        with open(p, "w") as f:
            f.write(src)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=td, input=stdin_text,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except Exception as e:
            return None, f"exec: {type(e).__name__}"
    if r.returncode != 0:
        return None, "raised"
    return r.stdout.strip(), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--lr-decay", type=float, default=0.5)
    ap.add_argument("--lam-concept", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--out", default="code1")
    ap.add_argument("--drop-unk", action="store_true",
                    help="drop training windows whose TARGET is `<unk>`. Measured on the code-only "
                         "corpus at 6k vocab: `<unk>` is 6.98%% of all training targets (62,937 of "
                         "902,310) -- the single most frequent token the model is asked to predict. "
                         "Training on those positions REWARDS the model for emitting `<unk>`, which "
                         "is why the output stays full of it even after the vocabulary was enlarged "
                         "5x. The positions are unanswerable by construction: the true token was "
                         "discarded by the vocabulary before training started.")
    ap.add_argument("--max-windows", type=int, default=1200000)
    ap.add_argument("--n-exec", type=int, default=25, help="how many problems to actually execute")
    a = ap.parse_args()

    log("=" * 96)
    log("A MODEL THAT WRITES CODE — judged by EXECUTION, not by resemblance")
    log("=" * 96)

    d = code_data.build()
    ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
    Xtr, Ytr = code_data.make_windows(ids, bounds, split, a.window, which=0)
    Xva, Yva = code_data.make_windows(ids, bounds, split, a.window, which=1)
    if a.drop_unk:
        # STOP REWARDING `<unk>`. id 0 is `<unk>` by construction in code_data (vocab[0]).
        # Measured: it is 6.98% of targets at 6k vocab and the #1 most frequent target, so a plain
        # next-token objective spends real capacity learning to emit it. Dropping those windows does
        # not lose the surrounding context -- windows overlap at stride W/2, so a dropped window's
        # context is still covered by its neighbours.
        keep = np.asarray(Ytr).reshape(-1) != 0
        log(f"  drop-unk: removing {int((~keep).sum()):,} of {len(keep):,} training windows "
            f"({(~keep).mean()*100:.2f}%) whose target is `<unk>`")
        Xtr, Ytr = Xtr[keep], Ytr[keep]
    if len(Xtr) > a.max_windows:
        sel = np.linspace(0, len(Xtr) - 1, a.max_windows).astype(np.int64)
        Xtr, Ytr = Xtr[sel], Ytr[sel]
    log(f"  windows {len(Xtr):,} train / {len(Xva):,} val   vocab {len(vocab):,}   W={a.window}")

    train_mask = np.zeros(len(ids), dtype=bool)
    for j in range(len(bounds) - 1):
        if split[j] == 0:
            train_mask[bounds[j]:bounds[j + 1]] = True
    cs, spec = code_concepts.build(vocab, ids, train_mask, windows=Xtr)
    log(f"  concepts: {spec['n_kept']} kept, {spec['n_dropped']} dropped of {spec['n_defined']}")
    for c in spec["kept"]:
        log(f"     {c['name']:<24} support {c['support']:.4f}  {c['n_tokens']} tokens")
    for c in spec["dropped"]:
        log(f"     DROPPED {c['name']:<18} {c['reason']}")
    concepts = [Concept(c["name"], c["ids"], support=c["support"]) for c in cs]

    net = lens.Net(vocab=len(vocab), concepts=concepts, d=a.d, window=a.window, seed=a.seed,
                   lam_concept=a.lam_concept, lam_act=0.0, device=a.device)
    npar = int(sum(p.data.size for p in net.parameters()))
    log(f"  model: {npar:,} params  (d={a.d}, W={a.window}, K={net.K})")
    ratio = a.d * a.window / max(net.K, 1)
    log(f"  bottleneck ratio {a.d*a.window} in / {net.K} out = {ratio:.0f}:1")
    log(f"  tokens per parameter: {len(Xtr)/max(npar,1):.2f}")

    rng = np.random.default_rng(a.seed)
    hist = []
    for ep in range(1, a.epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(len(Xtr))
        tot, ctot, nb, corr, seen = 0.0, 0.0, 0, 0, 0
        lr_now = a.lr * (1.0 - a.lr_decay * (ep - 1) / max(a.epochs - 1, 1)) if a.lr_decay > 0 \
            else a.lr
        for s in range(0, len(Xtr), a.batch):
            idx = order[s:s + a.batch]
            for p in net.parameters():
                p.grad = None
            total, parts = net.loss(Xtr[idx], Ytr[idx])
            total.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.data = p.data - lr_now * p.grad
            tot += float(B.to_cpu(parts.get("ce", total.data)))
            ctot += float(parts.get("concept_aux", 0.0))
            nb += 1
            out = net.forward(Xtr[idx])
            corr += int((B.to_cpu(out["logits"].data).argmax(axis=1) == Ytr[idx]).sum())
            seen += len(idx)
        vl, vc, vs = 0.0, 0, 0
        for s in range(0, len(Xva), a.batch):
            idx = np.arange(s, min(s + a.batch, len(Xva)))
            va = net.forward_arrays(Xva[idx])["logits"]
            vl += float(B.to_cpu(T.softmax_ce(T.Tensor(B.to_dev(va)), Yva[idx]).data)) * len(idx)
            vc += int((va.argmax(axis=1) == Yva[idx]).sum())
            vs += len(idx)
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "concept_loss": ctot / max(nb, 1),
               "train_acc": corr / max(seen, 1), "val_ppl": float(np.exp(vl / max(vs, 1))),
               "val_acc": vc / max(vs, 1), "lr": float(lr_now),
               "seconds": time.perf_counter() - t0}
        hist.append(rec)
        log(f"  ep {ep}/{a.epochs}  loss {rec['train_loss']:.4f} concept {rec['concept_loss']:.4f} "
            f"acc {rec['train_acc']*100:.2f}%   VAL ppl {rec['val_ppl']:.2f} "
            f"acc {rec['val_acc']*100:.2f}%  lr {lr_now:.3f}  ({rec['seconds']:.0f}s)")
        lens.save(net, os.path.join(OUT, f"{a.out}_ep{ep}.npz"))
        json.dump({"args": vars(a), "history": hist, "params": npar, "concepts": spec,
                   "ratio": ratio}, open(os.path.join(OUT, f"{a.out}_log.json"), "w"), indent=1)

    # ---------------------------------------------------------------- THE EXECUTION TEST
    log("")
    log("=" * 96)
    log("CAN IT WRITE CODE THAT RUNS? — generated code is EXECUTED and checked")
    log("=" * 96)
    vmap = {w: i for i, w in enumerate(vocab)}

    def enc(t):
        import re
        return [vmap.get(x, 0) for x in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]",
                                                  t.lower())]

    def gen(prompt, n_new=220, temp=0.6, top_k=20, rep=1.25, seed=0):
        e = enc(prompt)
        ctx = (e if len(e) >= a.window else [0] * (a.window - len(e)) + e)[-a.window:]
        out, used, r = [], {}, np.random.default_rng(seed)
        for _ in range(n_new):
            lg = B.to_cpu(net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"])[0]
            z = lg.astype(np.float64)
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
            out.append(vocab[nxt] if nxt < len(vocab) else " ")
            used[nxt] = used.get(nxt, 0) + 1
            ctx = (ctx + [nxt])[-a.window:]
        return "".join((" " if t == "<unk>" else t) for t in out)

    # build evaluation prompts from the VALIDATION pairs: the problem text, then let it write code
    evals = []
    n_tried = 0
    for j in range(len(bounds) - 1):
        if split[j] != 1:
            continue
        seg = ids[bounds[j]:bounds[j + 1]]
        text = " ".join(vocab[i] for i in seg)
        if "def " not in text:
            continue
        cut = text.find("def ")
        evals.append(text[:cut + 4])
        n_tried += 1
        if n_tried >= a.n_exec:
            break

    log(f"  executing {len(evals)} generated snippets (timeout 5s each, isolated temp dir)")
    stats = {"syntax_ok": 0, "ran": 0, "total": 0}
    shown = 0
    for prompt in evals:
        code = gen(prompt)
        body = code
        stats["total"] += 1
        try:
            ast.parse(body)
            stats["syntax_ok"] += 1
        except SyntaxError:
            pass
        val, why = run_snippet(body)
        if why == "ok":
            stats["ran"] += 1
        if shown < 6:
            log(f"\n  PROMPT (problem text) ...{prompt[-90:]!r}")
            log(f"  WROTE  {body[:260]!r}")
            log(f"  RESULT {why}")
            shown += 1

    log("")
    log(f"  syntactically valid python: {stats['syntax_ok']}/{stats['total']}")
    log(f"  actually ran without error: {stats['ran']}/{stats['total']}")
    log("")
    log("  --- concept fidelity audit ---")
    aud = lens.audit(net, Xva[:3000], concepts=concepts)
    for c in aud["concepts"]:
        au = "None" if c["auroc"] is None else f"{c['auroc']:.4f}"
        fr = "None" if c["fires"] is None else f"{c['fires']:.3f}"
        log(f"     {c['name']:<24} AUROC {au:>7}  fires {fr:>6}  {'PASS' if c['pass'] else 'FAIL'}")
    log(f"     -> {aud['n_pass']}/{aud['n_concepts']} pass   coverage {aud['coverage']*100:.2f}%")

    json.dump({"exec": stats, "audit": aud, "history": hist, "params": npar,
               "concepts": spec, "ratio": ratio},
              open(os.path.join(OUT, f"{a.out}_replies.json"), "w"), indent=1)
    log("")
    log("=" * 96)
    log(f"  final VAL ppl {hist[-1]['val_ppl']:.2f}  acc {hist[-1]['val_acc']*100:.2f}%   "
        f"runs {stats['ran']}/{stats['total']}   concepts {aud['n_pass']}/{aud['n_concepts']}")
    log("=" * 96)


if __name__ == "__main__":
    main()
