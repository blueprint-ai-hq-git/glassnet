"""lens/tests/test_l4.py — THE L4 GATE: the library.

The library is the deliverable, so the gate is written the way the library will be USED: build,
train, ask why, check the names, save and reload. If the usage is ugly, the gate should show it.

The most important test here is the DEAD CONCEPT test. The old project shipped a model at 100%
coverage whose concepts were nearly all dead -- coverage without an audit is worthless. So this
gate declares a concept that CANNOT be learned (its token never appears in the training text),
trains, and requires that `audit()` FAILS it. An audit that cannot fail anything is decoration.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} {detail}")


print("=" * 100)
print("L4 GATE — the library")
print("=" * 100)

import lens                                                # noqa: E402
from lens import backend as B                              # noqa: E402

print(f"lens exports: {', '.join(lens.__all__)}")

# ---------------------------------------------------------------- a small corpus
V, W, N = 40, 6, 5000
rng = np.random.default_rng(0)
X = rng.integers(0, 20, size=(N, W)).astype(np.int64)
# INTERLEAVED on purpose. The first version put contains_30 in X[:N//2] and contains_31 in
# X[::4], so the test split X[4000:] contained no instance of contains_30 at all and its audit
# had no positives -- a defect in the corpus, not the model. Interleaving guarantees any split
# contains both concepts, which is what a corpus used for auditing must do.
X[::2, 0] = 30                          # concept "contains_30" -- half the windows
X[1::2, 1] = 31                         # concept "contains_31" -- the other half
y = ((X[:, 0] == 30) * 5 + (X[:, 1] == 31) * 7 + X[:, 2] % 4).astype(np.int64) % V

train_X, train_y = X[:4000], y[:4000]
test_X, test_y = X[4000:], y[4000:]

# The audit sample MUST be shuffled. `test_X` is X[4000:], which is entirely the corpus's second
# half -- so contains_30 (which lives in the first half) has no positives there and cannot be
# measured. That is a defect in the test, not the model: the same mistake was made once before in
# the L3 gate, which is why it is called out here.
AUD = np.random.default_rng(99).permutation(len(test_X))[:1500]
audit_X = test_X[AUD]

# ---------------------------------------------------------------- 1. the 10-line path
print("\n  --- the 10-line usage path ---")
concepts = lens.concepts_from_words(train_X, [30, 31], window=W)
report("1. concepts_from_words works", len(concepts) == 2,
       ", ".join(f"{c.name} (support {c.support:.3f})" for c in concepts))

net = lens.Net(vocab=V, concepts=concepts, d=64, window=W, seed=0, device="cpu")
report("2. Net builds", net.K == 2 and net.W == W, f"K={net.K}, W={net.W}, d={net.D}")

res = lens.train(net, train_X, train_y, epochs=8, batch=128, lr=0.5, seed=1)
report("3. train runs and reports", len(res.history) == 8 and res.final_loss is not None,
       f"{res!r}")

# ---------------------------------------------------------------- 2. explain is exact
e = lens.explain(net, test_X[0], top_k=2)
report("4. explain() is EXACT", e["residual"] < 1e-4, f"residual {e['residual']:.3e}")
print("\n  --- what the library prints ---")
for line in lens.explain_text(net, test_X[0], top_k=2).splitlines():
    print("      " + line)

# ---------------------------------------------------------------- 3. the audit
a = lens.audit(net, audit_X)
report("5. audit() runs and reports coverage WITH it", a["n_concepts"] == 2 and a["coverage"] > 0,
       f"{a['n_pass']}/{a['n_concepts']} concepts pass, coverage {a['coverage']*100:.4f}%")
for c in a["concepts"]:
    au = "None" if c["auroc"] is None else f"{c['auroc']:.4f}"
    fr = "None" if c["fires"] is None else f"{c['fires']:.3f}"
    print(f"      {c['name']:<20} AUROC {au:<8} fires {fr:<8} "
          f"{'PASS' if c['pass'] else 'FAIL'}  {c['why']}")
report("6. the real concepts pass the audit", a["n_pass"] == 2, f"{a['n_pass']}/2")

# ---------------------------------------------------------------- 4. THE DEAD CONCEPT TEST
# A concept whose token never occurs in the training text. It cannot be learned, so the audit
# MUST fail it. This is the check the old project's coverage number could not make.
dead = lens.concepts_from_words(train_X, [30], window=W)[0]
dead.name = "contains_39_NEVER_IN_TRAIN"
dead.ids = [39]
dead.support = 0.5
net_dead = lens.Net(vocab=V, concepts=[concepts[0], concepts[1], dead], d=64, window=W,
                    seed=0, device="cpu")
lens.train(net_dead, train_X, train_y, epochs=4, batch=128, lr=0.5, seed=1)
a_dead = lens.audit(net_dead, audit_X)
dead_row = [c for c in a_dead["concepts"] if c["name"] == dead.name][0]
report("7. audit CATCHES a dead concept", dead_row["pass"] is False,
       f"{dead.name}: auroc {dead_row['auroc']} fires {dead_row['fires']} -> "
       f"FAIL ({dead_row['why']})")
report("7b. coverage is still high despite the dead concept",
       a_dead["coverage"] > 0.5,
       f"coverage {a_dead['coverage']*100:.4f}% with {a_dead['n_pass']}/{a_dead['n_concepts']} "
       f"concepts passing -- THIS is why coverage alone is not evidence")

# ---------------------------------------------------------------- 5. save / load
p = os.path.join(os.path.dirname(_HERE), "runs", "l4_roundtrip.npz")
os.makedirs(os.path.dirname(p), exist_ok=True)
lens.save(net, p)
net2 = lens.load(p)
worst = max(float(np.abs(B.to_cpu(x.data) - B.to_cpu(z.data)).max())
            for x, z in zip(net.parameters(), net2.parameters()))
report("8. save/load round-trips exactly", worst == 0.0, f"worst |diff| = {worst:.3e}")
report("8b. concept names survive the round trip",
       [c.name for c in net2.concepts] == [c.name for c in net.concepts],
       ", ".join(c.name for c in net2.concepts))
e2 = lens.explain(net2, test_X[0], top_k=2)
report("8c. the reloaded model still explains exactly", e2["residual"] < 1e-4,
       f"residual {e2['residual']:.3e}")

# ---------------------------------------------------------------- 6. the same library on the GPU
print("\n  --- the same 3 lines on the GPU ---")
try:
    netg = lens.Net(vocab=V, concepts=concepts, d=64, window=W, seed=0, device="gpu")
    resg = lens.train(netg, train_X, train_y, epochs=4, batch=128, lr=0.5, seed=1)
    eg = lens.explain(netg, test_X[0], top_k=2)
    ag = lens.audit(netg, audit_X)
    report("9. the same library works on the GPU", eg["residual"] < 1e-4,
           f"explain residual {eg['residual']:.3e}, audit {ag['n_pass']}/{ag['n_concepts']} pass")
except Exception as ex:
    report("9. the same library works on the GPU", False, f"{type(ex).__name__}: {ex}")
finally:
    lens.tensor.set_device("cpu")

print("\n" + "=" * 100)
npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
print(f"L4 GATE: {npass}/{len(RESULTS)} passed")
for nm, st, dd in RESULTS:
    if st != "PASS":
        print(f"  !! {nm}: {dd}")
print("=" * 100)
sys.exit(0 if npass == len(RESULTS) else 1)
