"""lens/tests/test_l3.py — THE L3 GATE: the readable layer.

What this gate insists on, because each of these was a real failure in the old project:

1. THE CONCEPT LOSS IS ACTIVE. `parts["concept_aux"]` must be present and non-zero. The old
   project never logged it, so a run that trained on the wrong concept set looked identical to
   one that did not.

2. ITS GRADIENT REACHES THE CONCEPT LAYER. Measured directly: zero the concept term, take the
   gradient, then enable it and take the gradient again. If the difference is zero, the concept
   supervision is doing nothing regardless of what the loss value says.

3. explain() IS EXACT. Contributions plus bias must sum to the model's own score. The residual
   is printed. An explanation you cannot check is a story.

4. COVERAGE IS 100% BY CONSTRUCTION -- and reported next to the audit, never alone, because the
   old project had a model at 100% coverage whose concepts were nearly all dead.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

RNG = np.random.default_rng(0)
RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<38} {detail}")


print("=" * 100)
print("L3 GATE — the readable concept layer, on our own engine")
print("=" * 100)

from lens.concepts import Concept, explain, explain_text, coverage   # noqa: E402
from lens.model import LensNet                                       # noqa: E402

# ---------------------------------------------------------------- a tiny synthetic corpus
# Deliberately simple so the gate is about the MECHANISM, not about whether a model can learn.
# Vocabulary: 0..9 plus a few marker tokens. Windows of 6. The next token is a fixed function of
# the window, so a correct model can learn it and the concepts have real support.
V = 16
W = 6
N = 4000
marker_a, marker_b = 10, 11
X = RNG.integers(0, 10, size=(N, W)).astype(np.int64)
half = N // 2
X[:half, 0] = marker_a                      # concept "has_marker_a"
X[half:, 0] = marker_b                      # concept "has_marker_b"
X[::3, 1] = 7                               # concept "seven_in_position_1"
y = ((X[:, 0] == marker_a) * 3 + (X[:, 1] == 7) * 5 + X[:, 2] % 4).astype(np.int64) % V

concepts = [
    Concept("has_marker_a", [marker_a], support=float((X[:, 0] == marker_a).mean())),
    Concept("has_marker_b", [marker_b], support=float((X[:, 0] == marker_b).mean())),
    Concept("seven_in_position_1", [7], support=float((X[:, 1] == 7).mean())),
]
report("concepts declared", True,
       ", ".join(f"{c.name} (support {c.support:.3f})" for c in concepts))

net = LensNet(V, concepts, d=32, window=W, seed=0, lam_concept=0.1, lam_act=0.0)
nparams = sum(p.data.size for p in net.parameters())
report("model builds", True, f"{nparams} parameters, K={net.K} named concepts")

# ---------------------------------------------------------------- 1. concept loss active
loss, parts = net.loss(X[:64], y[:64])
report("1. concept loss is ACTIVE", "concept_aux" in parts and parts["concept_aux"] > 0,
       f"parts = { {k: round(v, 5) for k, v in parts.items()} }")

# ---------------------------------------------------------------- 2. its gradient reaches h
net.zero_all = None
for p in net.parameters():
    p.grad = None
l_off, _ = net.loss(X[:64], y[:64], use_concept_loss=False)
l_off.backward()
g_off = net.concept.W.grad.copy()

for p in net.parameters():
    p.grad = None
l_on, _ = net.loss(X[:64], y[:64], use_concept_loss=True)
l_on.backward()
g_on = net.concept.W.grad.copy()

delta = float(np.abs(g_on - g_off).max())
report("2. concept loss gradient reaches the layer", delta > 0,
       f"max |grad_with - grad_without| = {delta:.3e} on concept.W")

# also: is the gradient non-zero in absolute terms?
report("2b. concept.W receives a non-zero gradient", float(np.abs(g_on).max()) > 0,
       f"max |grad| = {float(np.abs(g_on).max()):.3e}")

# ---------------------------------------------------------------- 3. train a little
opt_lr = 0.5
for step in range(300):
    for p in net.parameters():
        p.grad = None
    idx = RNG.permutation(N)[:128]
    l, _ = net.loss(X[idx], y[idx])
    l.backward()
    for p in net.parameters():
        if p.grad is not None:
            p.data = p.data - opt_lr * p.grad
    if step % 100 == 99:
        print(f"        step {step+1}: loss {float(l.data):.4f}")

# ---------------------------------------------------------------- 4. explain is EXACT
worst = 0.0
for i in range(200):
    e = explain(net, X[i], top_k=3)
    worst = max(worst, e["residual"])
report("3. explain() is EXACT", worst < 1e-4,
       f"worst |contributions + bias - model score| = {worst:.3e} over 200 windows")

e = explain(net, X[0], top_k=3)
print("\n        sample explanation:")
for line in explain_text(net, X[0], top_k=3).splitlines():
    print("        " + line)
print()

# ---------------------------------------------------------------- 5. coverage, with the audit
# The audit sample MUST be shuffled. Taking X[:500] samples only the corpus's first half, where
# marker_b never appears and marker_a always does -- so two of the three concepts had no valid
# audit at all ("no positives in this sample"). That is a defect in the test, not the model.
audit_idx = RNG.permutation(N)[:500]
Xa = X[audit_idx]
cov = coverage(net, Xa)
report("4. coverage is 100% (by construction)", abs(cov - 1.0) < 1e-9,
       f"coverage = {cov*100:.4f}%  -- reported WITH the audit below, never alone")

h = net.forward_arrays(Xa)["concepts"]
print("        audit of each concept against its OWN declared target (shuffled sample):")
npass = 0
for i, c in enumerate(concepts):
    tgt = c.labels(Xa)
    if tgt.sum() == 0 or (~tgt).sum() == 0:
        print(f"          {c.name:<24} no positives in this sample")
        continue
    score = h[:, i]
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(tgt.sum())
    au = (ranks[tgt].sum() - n1 * (n1 + 1) / 2) / (n1 * int((~tgt).sum()))
    fired = float((score > 0.5).mean())
    ok = au >= 0.55
    npass += int(ok)
    print(f"          {c.name:<24} AUROC {au:.4f}   fires {fired:.3f}   "
          f"{'PASS' if ok else 'FAIL'} (bar 0.55)")
report("4b. at least half the concepts pass", npass >= (len(concepts) + 1) // 2,
       f"{npass}/{len(concepts)} concepts pass AUROC >= 0.55")

print("\n" + "=" * 100)
npass_all = sum(1 for _, s, _ in RESULTS if s == "PASS")
print(f"L3 GATE: {npass_all}/{len(RESULTS)} passed")
for nm, st, dd in RESULTS:
    if st != "PASS":
        print(f"  !! {nm}: {dd}")
print("=" * 100)
sys.exit(0 if npass_all == len(RESULTS) else 1)
