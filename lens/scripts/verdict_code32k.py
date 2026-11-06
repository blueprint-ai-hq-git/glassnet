"""verdict_code32k.py — gates for the vocabulary experiment, using the AUTHORITATIVE harness.

WHY THIS REUSES reexec_code.py's EXACT construction rather than writing a new one:
  Today I compared 8/25 (reexec_code) against 0/25 (my own sweep) for the SAME checkpoint and got a
  false conclusion, because the two harnesses build prompts differently. Any new number must come
  from the SAME construction as the baseline it is compared against, or the comparison is worthless.

  So this script:
    * loads the checkpoint by env var (CKPT), default the new 32k model
    * builds prompts EXACTLY as reexec_code.py does (style A: cut at 'def '; style B: cut after the
      signature colon, from the full document text; plus the clean subset with no <unk>)
    * uses reexec_code's decoder settings (n_new 200, temp 0.5, top_k 15, rep 1.2)
    * reports the clean-subset number, because that is the only fair measure of body completion

GATES (pre-registered in run_code32k.sh before training)
  q1  syntax rate beats 4/25 (the authoritative baseline for code1_ep16, re-measured today)
  q2  <unk> share of generated tokens falls below the 6k baseline's
  q3  if q1 fails -> vocabulary was NOT the blocker; report it and stop adding vocabulary
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
NEW = os.environ.get("CKPT", "covocab32k_ep16")
OLD = "code1_ep16"
TOK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]")
BASELINE_SYNTAX = 4                                       # re-measured today, authoritative harness


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


def prompts_for(vocab, ids, bounds, split, n=25):
    """EXACTLY reexec_code.py's construction: A = cut at 'def ', B = cut after the sig colon."""
    pa, pb = [], []
    for j in range(len(bounds) - 1):
        if len(pa) >= n and len(pb) >= n:
            break
        if split[j] != 1:
            continue
        text = " ".join(vocab[i] for i in ids[bounds[j]:bounds[j + 1]])
        k = text.find("def ")
        if k < 0:
            continue
        if len(pa) < n:
            pa.append(text[:k + 4])
        colon = text.find(" : ", k)
        if colon > 0 and len(pb) < n:
            pb.append(text[:colon + 3])
    return pa, pb


print("=" * 100)
print(f"CODE VERDICT — {NEW} (vocab 32k, code-only) vs {OLD} (vocab 6k, both)")
print("=" * 100)

# EACH CHECKPOINT MUST BE EVALUATED WITH ITS OWN VOCABULARY.
# A first version of this script called code_data.build() once and used whatever vocab it returned
# (6000, from the default env), then fed that to the 32k checkpoint — so every token mapped to the
# WRONG WORD and the whole comparison was meaningless. The checkpoint's own V is now the source of
# truth, and a mismatch refuses instead of printing a wrong number.
def load_for(tag):
    f = os.path.join(R, tag + ".npz")
    if not os.path.exists(f):
        return None, None
    net = lens.load(f, device="cpu")
    cands = [p for p in os.listdir(os.path.join(PROJ, "data", "code", "prepared"))
             if p.startswith("code_vocab")]
    best = None
    for c in cands:
        v = json.load(open(os.path.join(PROJ, "data", "code", "prepared", c)))
        if len(v) == net.V:
            best = v
            break
    if best is None:
        print(f"  {tag}: NO VOCAB of width {net.V} found — refusing to name words across a mismatch")
    return net, best


# prompts are built per-checkpoint, because the token ids differ between vocabularies
def prompts_for_checkpoint(net, vocab, n=25):
    ci = os.path.join(PROJ, "data", "code", "prepared")
    tag = f"v{net.V}_{'code' if net.V > 6000 else 'both'}"
    ip = os.path.join(ci, f"code_ids_{tag}.npy")
    bp = os.path.join(ci, f"code_bounds_{tag}.npy")
    sp = os.path.join(ci, f"code_split_{tag}.npy")
    if not all(os.path.exists(p) for p in (ip, bp, sp)):
        tag = f"v{net.V}_code" if net.V > 6000 else "v6000_both"
        ip = os.path.join(ci, f"code_ids_{tag}.npy")
        bp = os.path.join(ci, f"code_bounds_{tag}.npy")
        sp = os.path.join(ci, f"code_split_{tag}.npy")
    ids = np.load(ip); bounds = np.load(bp); split = np.load(sp)
    pa, pb = [], []
    for j in range(len(bounds) - 1):
        if len(pa) >= n and len(pb) >= n:
            break
        if split[j] != 1:
            continue
        text = " ".join(vocab[i] for i in ids[bounds[j]:bounds[j + 1]])
        k = text.find("def ")
        if k < 0:
            continue
        if len(pa) < n:
            pa.append(text[:k + 4])
        colon = text.find(" : ", k)
        if colon > 0 and len(pb) < n:
            pb.append(text[:colon + 3])
    return pa, pb


def measure(tag):
    net, vocab = load_for(tag)
    if net is None or vocab is None:
        return None, None
    pa, pb = prompts_for_checkpoint(net, vocab)
    pb_clean = [p for p in pb if "<unk>" not in p]
    print(f"  {tag}: V={net.V:,} vocab words={len(vocab):,} prompts A {len(pa)} B {len(pb)} "
          f"B-clean {len(pb_clean)}")
    res = {}
    for label, prompts in (("A", pa), ("B", pb), ("B_clean", pb_clean)):
        syn = run = unk_tok = tot_tok = 0
        ex = None
        for i, pr in enumerate(prompts):
            toks, src = gen(net, vocab, pr, seed=i)
            unk_tok += sum(1 for t in toks if t == "<unk>")
            tot_tok += len(toks)
            ok = run_snippet(src)
            if ok is not None:
                syn += 1
                if ex is None:
                    ex = src[:160]
            if ok:
                run += 1
        res[label] = {"syntax": syn, "ran": run, "total": len(prompts),
                      "unk": unk_tok / max(tot_tok, 1), "example": ex}
    return res, vocab

new, _ = measure(NEW)
old, _ = measure(OLD)

if new is None:
    print(f"  {NEW} not found — verdict WITHHELD (not guessed)")
    sys.exit(0)

print(f"\n  {'model':<16}{'style':<9}{'syntax':>9}{'ran':>6}{'unk in output':>15}")
for tag, res in ((OLD, old), (NEW, new)):
    if not res:
        continue
    for lab in ("A", "B", "B_clean"):
        r = res[lab]
        print(f"  {tag:<16}{lab:<9}{r['syntax']:>4}/{r['total']:<4}{r['ran']:>6}"
              f"{r['unk']*100:>14.1f}%")

q1 = new["B"]["syntax"] > BASELINE_SYNTAX
q2 = (new["B"]["unk"] < old["B"]["unk"]) if old else None
print("\n" + "-" * 100)
print("  GATES — compared values beside verdicts")
print(f"    q1  style-B syntax {new['B']['syntax']}/{new['B']['total']} > baseline "
      f"{BASELINE_SYNTAX}/{BASELINE_SYNTAX and 25} -> {'PASS' if q1 else 'FAIL'}")
if q2 is not None:
    print(f"    q2  <unk> in output {new['B']['unk']*100:.1f}% < {old['B']['unk']*100:.1f}% "
          f"-> {'PASS' if q2 else 'FAIL'}")
print(f"    (clean-prompt subset — the fair body-completion measure: "
      f"{new['B_clean']['syntax']}/{new['B_clean']['total']} new vs "
      f"{old['B_clean']['syntax'] if old else '?'}/{old['B_clean']['total'] if old else '?'} old)")
if q1:
    verdict = (f"VOCABULARY WAS A REAL BLOCKER: 32k vocab raises style-B syntax from "
               f"{BASELINE_SYNTAX}/25 to {new['B']['syntax']}/25.")
else:
    verdict = ("q1 FAILS: the 5.3x larger vocabulary did NOT raise the syntax rate -> vocabulary was "
               "NOT the blocker. Report it and stop adding vocabulary; the remaining cause is the "
               "model's capacity/data, not its word list.")
print(f"\n  VERDICT: {verdict}")
json.dump({"new": new, "old": old, "gates": {"q1": bool(q1), "q2": q2}, "verdict": verdict},
          open(os.path.join(R, "code32k_verdict.json"), "w"), indent=1)
print(f"  saved {os.path.join(R, 'code32k_verdict.json')}")
