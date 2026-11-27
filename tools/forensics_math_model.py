"""LOOK INSIDE THE MATHS MODEL. Not another training run - an autopsy of the one we have.

Questions this answers, all from the model's own weights and real held-out data:
  1. Which of its 29 named concepts are ALIVE on maths text, and do they mean what their names say?
  2. How much of the answer comes from the concept route versus the direct route that bypasses it?
  3. Why do runs "get randomly doomed" - is the concept layer saturating, or the logits exploding?
  4. What does it attend to when it writes a wrong number?
"""
import json
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ + "/lens/scripts")
import chat_data as cd
import chat_concepts_wide as ccw

CK = sys.argv[1] if len(sys.argv) > 1 else PROJ + "/lens/runs/chat/math_explain_v2_ep3.npz"
DATA = PROJ + "/data/math/prepared_v2"
NSAMP = int(os.environ.get("FOR_N", "4000"))

vocab = json.load(open(DATA + "/vocab_v12000.json"))
z = np.load(CK, allow_pickle=True)
E, cW, cb, aW, ab, dmix = (z["emb"], z["concept.W"], z["concept.b"], z["answer.W"], z["answer.b"], z["dmix"])
V, D = E.shape
W = cW.shape[0] // D
K = cW.shape[1]
print("checkpoint %s | V=%d D=%d window=%d concepts=%d" % (os.path.basename(CK), V, D, W, K))

ids = np.load(DATA + "/ids_v12000.npy", mmap_mode="r")
bounds = np.load(DATA + "/bounds_v12000.npy")
split = np.load(DATA + "/split_v12000.npy")
val_bounds = [int(b) for b, s in zip(bounds, split) if s == 1] + [int(bounds[-1])]
print("held-out conversations available: %d" % (len(val_bounds) - 1))

# windows drawn ONLY from held-out conversations
rng = np.random.default_rng(20261127)
starts = []
for i in range(len(val_bounds) - 1):
    b0, b1 = val_bounds[i], val_bounds[i + 1]
    if b1 - b0 > W + 2:
        starts.append(rng.integers(b0, b1 - W - 1))
starts = np.asarray(starts[:NSAMP])
X = np.stack([np.asarray(ids[s:s + W], dtype=np.int64) for s in starts])
Y = np.asarray([int(ids[s + W]) for s in starts])
print("windows: %d (all from held-out conversations)" % len(X))

spec = json.load(open(PROJ + "/lens/runs/chat/math_explain_v2_log.json"))["concepts"]
vmap = {w: i for i, w in enumerate(vocab)}
concepts = []
for k in spec["kept"]:
    tid = [vmap[w] for w in k.get("words", []) if w in vmap]
    if tid:
        concepts.append({"name": k["name"], "ids": tid})
concepts = concepts[:K]
print("concepts from the RUN's own spec: %d (checkpoint has %d columns)" % (len(concepts), K))
assert len(concepts) == K, "concept count still does not match the checkpoint"

flat = E[X].reshape(len(X), W * D)
pre = flat @ cW + cb
C = 1.0 / (1.0 + np.exp(-pre))          # activations, exactly as the model computes them
logits_c = C @ aW + ab
pre_d = flat @ dmix
logits_d = np.maximum(pre_d, 0) @ E.T
logits = logits_c + logits_d


def auroc(score, label):
    label = np.asarray(label).astype(bool)
    p, n = int(label.sum()), int((~label).sum())
    if p == 0 or n == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label].sum() - p * (p + 1) / 2.0) / (p * n))


print()
print("=== 1. ARE THE 29 CONCEPTS ALIVE, AND DO THEY MEAN THEIR NAMES? ===")
print("%-28s %9s %9s %9s %8s %s" % ("concept", "fires", "variance", "its-name", "AUROC", "verdict"))
rows = []
for j, c in enumerate(concepts):
    a = C[:, j]
    tgt = np.isin(X, c["ids"]).any(axis=1).astype(int)
    a_auroc = auroc(a, tgt)          # does the unit fire where its own words appear
    fires = float((a > 0.5).mean())
    var = float(a.var())
    if var < 1e-5:
        verdict = "DEAD"
    elif not isinstance(a_auroc, float) or np.isnan(a_auroc):
        verdict = "nameless"
    elif a_auroc >= 0.9:
        verdict = "honest"
    elif a_auroc >= 0.7:
        verdict = "weak"
    else:
        verdict = "MISLABELLED"
    rows.append({"concept": c["name"], "fires": fires, "var": var, "auroc": a_auroc, "verdict": verdict})
    print("%-28s %8.1f%% %9.5f %8.1f%% %8.3f %s" % (c["name"], 100 * fires, var, 100 * tgt.mean(), a_auroc, verdict))

dead = sum(1 for r in rows if r["verdict"] == "DEAD")
honest = sum(1 for r in rows if r["verdict"] == "honest")
mis = sum(1 for r in rows if r["verdict"] == "MISLABELLED")
print()
print("DEAD: %d   HONEST: %d   MISLABELLED: %d   (of %d)" % (dead, honest, mis, len(rows)))

print()
print("=== 2. WHICH ROUTE ACTUALLY DECIDES THE ANSWER? ===")
nc = float(np.linalg.norm(logits_c, axis=1).mean())
nd = float(np.linalg.norm(logits_d, axis=1).mean())
print("mean logit norm from the CONCEPT route: %.3f" % nc)
print("mean logit norm from the DIRECT route : %.3f" % nd)
print("direct route share: %.1f%%" % (100 * nd / (nc + nd)))
agree = float((logits_c.argmax(1) == logits.argmax(1)).mean())
print("how often the concept route alone picks the SAME next token as the full model: %.1f%%" % (100 * agree))

print()
print("=== 3. WHY RUNS GET RANDOMLY DOOMED ===")
print("activations pinned at 0 (sigmoid saturated low): %.1f%%" % (100 * float((C < 0.01).mean())))
print("activations pinned at 1 (sigmoid saturated high): %.1f%%" % (100 * float((C > 0.99).mean())))
print("max |logit| over all windows: %.2f   mean |logit|: %.2f" % (float(np.abs(logits).max()), float(np.abs(logits).mean())))
top1 = logits.argmax(1)
print("distinct next-token predictions on %d windows: %d" % (len(X), len(set(top1.tolist()))))
print("accuracy of the model on these held-out windows: %.2f%%" % (100 * float((top1 == Y).mean())))
print()
print("=== 4. WHAT DOES IT DO ON THE TOKENS THAT ARE NUMBERS? ===")
num_mask = np.array([vocab[int(t)].lstrip("-").replace(".", "").isdigit() for t in Y])
if num_mask.sum() > 20:
    print("windows whose true next token is a NUMBER: %d" % int(num_mask.sum()))
    print("  model gets them right              : %.2f%%" % (100 * float((top1[num_mask] == Y[num_mask]).mean())))
    print("  model predicts ANY number there    : %.2f%%"
          % (100 * float(np.mean([vocab[int(t)].lstrip("-").replace(".", "").isdigit() for t in top1[num_mask]]))))
    print("  model just repeats the previous token: %.2f%%"
          % (100 * float((top1[num_mask] == X[num_mask, -1]).mean())))

out = {"checkpoint": os.path.basename(CK), "windows": int(len(X)), "concepts": rows,
       "dead": dead, "honest": honest, "mislabelled": mis,
       "concept_route_norm": nc, "direct_route_norm": nd,
       "concept_route_agreement": agree, "next_token_acc": float((top1 == Y).mean())}
json.dump(out, open(PROJ + "/lens/runs/FORENSICS-MATH.json", "w"), indent=1)
print()
print("saved lens/runs/FORENSICS-MATH.json")
