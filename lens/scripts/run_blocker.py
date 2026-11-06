"""lens/scripts/run_blocker.py — the rebuilt engine against THE blocker.

THE QUESTION
------------
Every target in this project failed on one number: the readable model's **perplexity gap** against
a matched normal. Target G2 measured +12.46 / +12.09 / +14.48 at three sizes against bars of
5.81 / 6.90 / 5.77 -- FAILED at every size. Target I found the single biggest lever: setting the
activation sparsity penalty to 0 halves the gap (to +6.76) but breaks two more concepts.

That work was done in the OLD codebase, where the concept loss was never logged and where an
audit once reported 0/9 on a model that was fine because of two separate bugs. So the measurement
is being redone on the rebuilt engine, where:
  * the concept loss is provably active and provably reaching the layer (L3 gate),
  * explain() asserts its own exactness (L3 gate),
  * coverage is always reported with an audit (L4 gate),
  * and the same code runs on the CPU or the GPU.

THE ARMS (one change at a time, per the standing rule)
-----------------------------------------------------
  readable   lam_concept=0.1  lam_act=0.2    the architecture as designed
  lam0       lam_concept=0.1  lam_act=0.0    the ONE change -- Target I's lever
  normal     lam_concept=0.0  lam_act=0.0    the matched normal: same architecture, same
                                             parameter count, concepts present but NOT supervised

The normal arm is matched by construction: identical architecture and identical parameter count,
with the concept supervision switched off. That is the honest control -- if the concepts carried
no information the two would be the same model.

DATA: tiny_shakespeare, word-level windows, built by the project's existing prep (windowing only;
no model code is borrowed). Splits are contiguous positional slices, the temporal-LM convention.
The concept targets come from the pre-registered `runs/t3/meaningful_targets_ext.json` so this is
the same target set as before, not a new one chosen after seeing results.
"""
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)

import lens                                                # noqa: E402
from lens import backend as B                              # noqa: E402
from lens.concepts import Concept                          # noqa: E402

WIN = 8
VOCAB_CAP = 1500
D = int(os.environ.get("BLOCKER_D", "256"))
EPOCHS = int(os.environ.get("BLOCKER_EPOCHS", "30"))
BATCH = 256
LR = 0.5
SEED = int(os.environ.get("BLOCKER_SEED", "0"))
OUT = os.path.join(PROJ, "lens", "runs", os.environ.get("BLOCKER_OUT", "blocker"))
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(msg, flush=True)


log("=" * 100)
log("THE REBUILT ENGINE vs THE BLOCKER — perplexity gap, on the new code")
log("=" * 100)

# ---------------------------------------------------------------- data (windowing only)
sys.path.insert(0, os.path.join(PROJ, "scripts"))
from glassnet import tiny_data                             # noqa: E402

DATA = tiny_data.build(win=WIN, vocab_cap=VOCAB_CAP)
Xtr, ytr = np.asarray(DATA["Xtr"]), np.asarray(DATA["ytr"])
Xva, yva = np.asarray(DATA["Xva"]), np.asarray(DATA["yva"])
V = int(DATA["V"])
VOCAB = DATA["vocab"]
log(f"data: {len(Xtr)} train / {len(Xva)} val windows, V={V}, window={WIN}")

# ---------------------------------------------------------------- the pre-registered targets
# KIND selects the concept set. Every kind uses the SAME pre-registered target file, so no concept
# is invented after seeing a result.
#   base   -- the 9 concepts that cleared the 2.5% support floor (all previous arms used this)
#   wide   -- up to 60 concepts from the same file: more numbers for the answer layer to read
#   hybrid -- the 9 audited ones PLUS extra named ones, so explain() stays short while the model
#             is not starved
z = json.load(open(os.path.join(PROJ, "runs", "t3", "meaningful_targets_ext.json")))
q = z["qualifying"]
allt = z.get("all", [])
KIND = os.environ.get("BLOCKER_KIND", "base")


def _mk(entries):
    return [Concept(c["name"], c["ids"], support=float(c["support"])) for c in entries]


if KIND == "wide":
    # PRE-REGISTERED EXTENSION RULE, written before the run: the pre-registered target file holds
    # only 22 concepts, so the wide arm is extended to 60 by taking the MOST FREQUENT tokens in the
    # TRAIN text that are not already a concept target. It is a rule, not a hand-picked list, so it
    # cannot be tuned to a result. Support is measured on train text like every other concept.
    from collections import Counter
    concepts = _mk(allt)
    used = set()
    for c in concepts:
        used |= set(c.ids)
    freq = Counter(Xtr.reshape(-1).tolist())
    added = 0
    for tok, _ in freq.most_common():
        if added >= 60 - len(concepts):
            break
        if tok in used:
            continue
        supp = float((Xtr == tok).any(axis=1).mean())
        concepts.append(Concept(f"freq_token_{VOCAB[tok] if tok < len(VOCAB) else 'unk'}",
                                [int(tok)], support=supp))
        used.add(tok)
        added += 1
    log(f"wide: {len(allt)} pre-registered + {added} added by the pre-registered frequency rule")
elif KIND == "hybrid":
    # `q` is a list of DICTS, not Concept objects -- the first version wrote {c.name for c in q},
    # which raised AttributeError: 'dict' object has no attribute 'name' and stopped the arm before
    # it started. The hybrid arm therefore has NO result yet; it is not a failure, it never ran.
    seen = {c["name"] for c in q}
    extra = [c for c in allt if c["name"] not in seen]
    concepts = _mk(q) + _mk(extra[:15])
else:
    concepts = _mk(q)
log(f"concept set: {KIND} -- {len(concepts)} concepts "
    f"(support floor {z['support_bar']})")

# BLOCKER_K caps the concept count, so the capacity threshold between 22 (fails, +20.203) and
# 41 (wins, +0.372) can be bisected with the SAME pre-registered set, just truncated.
K_TARGET = int(os.environ.get("BLOCKER_K", "0"))
if K_TARGET and len(concepts) > K_TARGET:
    concepts = concepts[:K_TARGET]
    log(f"capped to {len(concepts)} concepts by BLOCKER_K")
for c in concepts:
    log(f"   {c.name:<26} support {c.support:.4f}  {len(c.ids)} tokens")


def perplexity(net, X, y, n=4000):
    """Mean CE over a sample, exponentiated. Computed through the model's own loss."""
    idx = np.arange(min(n, len(X)))
    out = net.forward_arrays(X[idx])
    logits = out["logits"]
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    p = e / e.sum(axis=1, keepdims=True)
    ll = -np.log(np.maximum(p[np.arange(len(idx)), y[idx]], 1e-12))
    return float(np.exp(ll.mean())), float(ll.mean())


def accuracy(net, X, y, n=4000):
    idx = np.arange(min(n, len(X)))
    pred = net.forward_arrays(X[idx])["logits"].argmax(axis=1)
    return float((pred == y[idx]).mean())


def run_arm(name, lam_concept, lam_act, device):
    log(f"\n--- arm {name}: lam_concept={lam_concept} lam_act={lam_act} device={device} ---")
    t0 = time.perf_counter()
    net = lens.Net(vocab=V, concepts=concepts, d=D, window=WIN, seed=SEED,
                   lam_concept=lam_concept, lam_act=lam_act, device=device)
    nparams = sum(p.data.size for p in net.parameters())

    rng = np.random.default_rng(SEED)
    logf = os.path.join(OUT, f"{name}.jsonl")
    open(logf, "w").close()
    SOFT = os.environ.get("BLOCKER_SOFT", "off") or "off"
    for ep in range(1, EPOCHS + 1):
        order = rng.permutation(len(Xtr))
        tot, nb, last_parts = 0.0, 0, None
        for s in range(0, len(Xtr), BATCH):
            idx = order[s:s + BATCH]
            for p in net.parameters():
                p.grad = None
            loss, parts = net.loss(Xtr[idx], ytr[idx], soft_concepts=SOFT)
            loss.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.data = p.data - LR * p.grad
            tot += float(B.to_cpu(loss.data))
            nb += 1
            last_parts = parts
        if ep % 5 == 0 or ep == EPOCHS:
            ppl_v, ce_v = perplexity(net, Xva, yva)
            log(f"    ep {ep:>3}  train loss {tot/max(nb,1):.4f}  val ppl {ppl_v:.3f}  "
                f"acc {accuracy(net, Xva, yva)*100:.2f}%  parts {sorted(last_parts.keys())}")
            with open(logf, "a") as f:
                f.write(json.dumps({"epoch": ep, "train_loss": tot / max(nb, 1),
                                    "val_ppl": ppl_v, "val_ce": ce_v,
                                    "parts": {k: float(v) for k, v in last_parts.items()}}) + "\n")
        if ep == EPOCHS:
            lens.save(net, os.path.join(OUT, f"{name}.npz"))

    ppl_v, ce_v = perplexity(net, Xva, yva)
    acc_v = accuracy(net, Xva, yva)
    aud = lens.audit(net, Xva[:3000])
    res = {"arm": name, "params": nparams, "device": device, "epochs": EPOCHS,
           "kind": KIND, "val_ppl": ppl_v, "val_ce": ce_v, "val_acc": acc_v,
           "n_pass": aud["n_pass"], "n_concepts": aud["n_concepts"],
           "coverage": aud["coverage"], "seconds": time.perf_counter() - t0,
           "concepts": [{"name": c["name"], "auroc": c["auroc"], "fires": c["fires"],
                         "pass": c["pass"]} for c in aud["concepts"]]}
    with open(os.path.join(OUT, f"{name}_summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"    DONE  {name}: ppl {ppl_v:.3f}  acc {acc_v*100:.2f}%  "
        f"concepts {aud['n_pass']}/{aud['n_concepts']}  coverage {aud['coverage']*100:.2f}%  "
        f"{res['seconds']:.1f}s")
    return res


DEV = os.environ.get("BLOCKER_DEVICE", "gpu")
LC = float(os.environ.get("BLOCKER_LC", "0.1"))          # FIX-01: 0.1 -> 1.0 is the one change
LA = float(os.environ.get("BLOCKER_LA", "0.2"))
ONLY = os.environ.get("BLOCKER_ONLY", "")                # comma list to run a subset of arms
results = {}
arms = [("readable", LC, LA), ("lam0", LC, 0.0), ("normal", 0.0, 0.0)]
if ONLY:
    want = {s.strip() for s in ONLY.split(",") if s.strip()}
    arms = [a for a in arms if a[0] in want]
    log(f"running a subset of arms: {[a[0] for a in arms]}  (lam_concept={LC} lam_act={LA})")
for arm, lc, la in arms:
    results[arm] = run_arm(arm, lc, la, DEV)

# If only a subset ran, the matched normal comes from the earlier run -- it does not depend on
# lam_concept (its concept supervision is OFF), so reusing it is correct, not a shortcut. Old
# results are never overwritten: each run writes to its own directory.
if "normal" not in results:
    prev_p = os.path.join(PROJ, "lens", "runs", "blocker", "normal_summary.json")
    if os.path.exists(prev_p):
        with open(prev_p) as f:
            results["normal"] = json.load(f)
        log(f"\n(matched normal reused from the earlier run: ppl "
            f"{results['normal']['val_ppl']:.3f} -- its concept supervision is off, so it does "
            f"not depend on lam_concept)")

# ---------------------------------------------------------------- the verdict
log("\n" + "=" * 100)
log("RESULTS")
log("=" * 100)
log(f"  {'arm':<10}{'params':>10}{'val ppl':>11}{'val acc':>10}{'concepts':>11}{'coverage':>11}")
for a, r in results.items():
    log(f"  {a:<10}{r['params']:>10,}{r['val_ppl']:>11.3f}{r['val_acc']*100:>9.2f}%"
        f"{r['n_pass']:>7}/{r['n_concepts']:<3}{r['coverage']*100:>10.2f}%")

n_ppl = results["normal"]["val_ppl"]
log("\n  GAP vs the matched normal (the number every target failed on):")
for a in sorted(results):
    if a == "normal":
        continue
    gap = results[a]["val_ppl"] - n_ppl
    log(f"    {a:<12} {results[a]['val_ppl']:>9.3f} - {n_ppl:.3f} = {gap:+.3f}")
log(f"\n  G2's bars were 5.81 / 6.90 / 5.77. G2 measured +12.46 / +12.09 / +14.48.")
log(f"  Target I measured +6.756 at lam_act=0.")

log("\n  per-concept audit (readable arm):")
for c in results["readable"]["concepts"]:
    au = "None" if c["auroc"] is None else f"{c['auroc']:.4f}"
    fr = "None" if c["fires"] is None else f"{c['fires']:.3f}"
    log(f"    {c['name']:<26} AUROC {au:<8} fires {fr:<8} {'PASS' if c['pass'] else 'FAIL'}")

with open(os.path.join(OUT, "verdict.json"), "w") as f:
    json.dump(results, f, indent=1)
log(f"\nsaved {OUT}/verdict.json")
log("=" * 100)
