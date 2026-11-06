"""lens/scripts/run_twophase.py — the structural fix: stop the two objectives fighting.

WHY THIS EXISTS
---------------
The sweep produced a clean, monotonic curve:

    lam_concept 0.1 -> 7/9 concepts, gap +12.281
    lam_concept 0.3 -> 9/9 concepts, gap +23.045
    lam_concept 0.5 -> 9/9 concepts, gap +29.216
    lam_concept 1.0 -> 9/9 concepts, gap +38.025

The cost rises smoothly with concept supervision strength. That is the signature of a cost that
lives in the SUPERVISION ITSELF, not in a setting -- so tuning cannot remove it.

THE OBSERVATION THAT MAKES THE FIX POSSIBLE
-------------------------------------------
`Concept.labels(X)` is a pure function of the text. The concept targets do not need a model. So
the concept layer does not have to be learned through the language loss at all -- it can be
trained DIRECTLY on its own targets, and then frozen.

    Phase A:  train embedding + concept layer on the concept targets ONLY. No language modelling.
    Phase B:  freeze those. Train ONLY the answer layer on next-token prediction. No concept loss.

Two objectives fighting over the same parameters become one objective each, on different
parameters. If the gap comes from that conflict, it should largely disappear.

WHAT THIS DOES NOT DO
---------------------
It does not add capability. Fixing the concepts means the answer layer cannot adapt them, so
accuracy may fall. That is stated as a prediction before the run, not discovered afterwards.

PRE-REGISTERED PREDICTIONS (written before running)
---------------------------------------------------
1. The perplexity gap falls WELL BELOW +12.281 -- because phase B has nothing competing with
   language modelling.
2. Accuracy may DROP below 29.65%, because the answer layer sees fixed concepts.
3. The concepts still pass 9/9, since phase A trains them directly on their own targets.

Bars unchanged: AUROC 0.55, firing floor 2%, gap against the matched normal (77.804).
"""
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import lens                                                # noqa: E402
from lens import backend as B                              # noqa: E402
from lens.concepts import Concept                          # noqa: E402
from glassnet import tiny_data                             # noqa: E402

WIN, VOCAB_CAP, D = 8, 1500, 256
EP_A = int(os.environ.get("TP_EPOCHS_A", "20"))
EP_B = int(os.environ.get("TP_EPOCHS_B", "40"))
BATCH, LR, SEED = 256, 0.5, 0
OUT = os.path.join(PROJ, "lens", "runs", "twophase")
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("TP_DEVICE", "gpu")


def log(m):
    print(m, flush=True)


log("=" * 100)
log("TWO-PHASE: concept layer trained directly on its own targets, then frozen")
log("=" * 100)

DATA = tiny_data.build(win=WIN, vocab_cap=VOCAB_CAP)
Xtr, ytr = np.asarray(DATA["Xtr"]), np.asarray(DATA["ytr"])
Xva, yva = np.asarray(DATA["Xva"]), np.asarray(DATA["yva"])
V, VOCAB = int(DATA["V"]), DATA["vocab"]
z = json.load(open(os.path.join(PROJ, "runs", "t3", "meaningful_targets_ext.json")))
concepts = [Concept(c["name"], c["ids"], support=float(c["support"])) for c in z["qualifying"]]
log(f"{len(Xtr)} train / {len(Xva)} val windows, V={V}, {len(concepts)} concepts, device={DEV}")

# ---------------------------------------------------------------- build the model
# lam_concept=0 for phase B: the concepts are already fixed, so there is nothing to supervise.
net = lens.Net(vocab=V, concepts=concepts, d=D, window=WIN, seed=SEED,
               lam_concept=0.0, lam_act=0.0, device=DEV)
nparams = sum(p.data.size for p in net.parameters())
log(f"model: {nparams:,} parameters")


def concept_targets(X):
    return np.stack([c.labels(X) for c in concepts], axis=1).astype(np.float32)


# ================================================================ PHASE A
log("\n--- PHASE A: train embedding + concept layer on the concept targets ONLY ---")
t0 = time.perf_counter()
concept_params = [net.emb, net.concept.W, net.concept.b]
rng = np.random.default_rng(SEED)
for ep in range(1, EP_A + 1):
    order = rng.permutation(len(Xtr))
    tot, nb = 0.0, 0
    for s in range(0, len(Xtr), BATCH):
        idx = order[s:s + BATCH]
        for p in concept_params:
            p.grad = None
        h = net.forward(Xtr[idx])["concepts"]          # (N, K)
        tgt = B.to_dev(concept_targets(Xtr[idx]))
        hd = h.data
        eps = 1e-7
        w = B.to_dev(net.pos_weight[None, :])
        bce = -(w * tgt * B.xp.log(hd + eps) + (1.0 - tgt) * B.xp.log(1.0 - hd + eps))
        loss = B.xp.asarray(bce.mean(), dtype=B.xp.float32)
        import lens.tensor as T
        node = T.Tensor(loss, h.requires_grad, _parents=(h,), _op="concept_bce",
                        _saved={"hd": hd, "tgt": tgt, "w": w, "n": bce.size})
        node.backward()
        for p in concept_params:
            if p.grad is not None:
                p.data = p.data - LR * p.grad
        tot += float(B.to_cpu(loss))
        nb += 1
    if ep % 5 == 0 or ep == 1:
        log(f"    ep {ep:>3}  concept BCE {tot/max(nb,1):.5f}")
log(f"    phase A done in {time.perf_counter()-t0:.1f}s")

# freeze
for p in concept_params:
    p.requires_grad = False
log("    embedding + concept layer FROZEN (requires_grad=False)")

# ================================================================ PHASE B
log("\n--- PHASE B: train ONLY the answer layer on next-token prediction ---")
t1 = time.perf_counter()
ans_params = net.answer.parameters()
rng = np.random.default_rng(SEED + 1)
for ep in range(1, EP_B + 1):
    order = rng.permutation(len(Xtr))
    tot, nb = 0.0, 0
    for s in range(0, len(Xtr), BATCH):
        idx = order[s:s + BATCH]
        for p in ans_params:
            p.grad = None
        loss, _ = net.loss(Xtr[idx], ytr[idx], use_concept_loss=False)
        loss.backward()
        for p in ans_params:
            if p.grad is not None:
                p.data = p.data - LR * p.grad
        tot += float(B.to_cpu(loss.data))
        nb += 1
    if ep % 10 == 0 or ep == EP_B:
        log(f"    ep {ep:>3}  CE {tot/max(nb,1):.4f}")
log(f"    phase B done in {time.perf_counter()-t1:.1f}s")

# ---------------------------------------------------------------- measure
idx = np.arange(min(4000, len(Xva)))
lg = net.forward_arrays(Xva[idx])["logits"]
lg = lg - lg.max(axis=1, keepdims=True)
e = np.exp(lg)
p = e / e.sum(axis=1, keepdims=True)
ppl = float(np.exp(-np.log(np.maximum(p[np.arange(len(idx)), yva[idx]], 1e-12)).mean()))
acc = float((p.argmax(axis=1) == yva[idx]).mean())
aud = lens.audit(net, Xva[:3000])

log("\n" + "=" * 100)
log("RESULTS")
log("=" * 100)
log(f"  two-phase   ppl {ppl:.3f}   acc {acc*100:.2f}%   "
    f"concepts {aud['n_pass']}/{aud['n_concepts']}   coverage {aud['coverage']*100:.2f}%")
log(f"  matched normal ppl 77.804   ->   GAP {ppl - 77.804:+.3f}")
log(f"  (for comparison: lc=0.1 7/9 gap +12.281 | lc=0.3 9/9 gap +23.045)")
log("\n  per-concept audit:")
for c in aud["concepts"]:
    au = "None" if c["auroc"] is None else f"{c['auroc']:.4f}"
    fr = "None" if c["fires"] is None else f"{c['fires']:.3f}"
    log(f"    {c['name']:<26} AUROC {au:<8} fires {fr:<8} {'PASS' if c['pass'] else 'FAIL'}")

res = {"arm": "twophase", "params": nparams, "val_ppl": ppl, "val_acc": acc,
       "n_pass": aud["n_pass"], "n_concepts": aud["n_concepts"], "coverage": aud["coverage"],
       "gap": ppl - 77.804, "epochs_a": EP_A, "epochs_b": EP_B,
       "concepts": [{"name": c["name"], "auroc": c["auroc"], "fires": c["fires"],
                     "pass": c["pass"]} for c in aud["concepts"]]}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(res, f, indent=1)
lens.save(net, os.path.join(OUT, "twophase.npz"))
log(f"\nsaved {OUT}/summary.json and twophase.npz")
log("=" * 100)
