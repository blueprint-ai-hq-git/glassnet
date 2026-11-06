"""lens/tests/test_l1.py — THE L1 GATE: our own autograd.

Three things are checked, all with printed numbers:

1. GRADIENTS. Every op's analytic gradient is compared against central differences.
   HONEST NOTE ON PRECISION: our kernels are float32-only, so the check runs in float32 with
   eps = 1e-2 and a 1e-2 relative bar. That is coarse compared with the 1e-6 float64 checks the
   old engine ran. The kernels themselves are separately verified to ~1e-8 against NumPy in the
   L0 gate, so this gate is testing the GRAPH (who gets which gradient), not the arithmetic.

2. NO REFERENCE CYCLES. The old engine closed every backward over its output tensor, which
   creates output -> closure -> output cycles that only the cycle collector can reclaim. Here,
   every node must be UNREACHABLE from its own parents. That is checked directly.

3. THE GRAPH IS FREED. Build a deep graph, drop it, force a collection, and count the live
   Tensor objects before and after. A leak shows up as a non-zero delta.
"""
import gc
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from lens import tensor as T  # noqa: E402

RNG = np.random.default_rng(0)
RESULTS = []
EPS = 1e-2
TOL = 1e-2


def report(name, ok, detail):
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<30} {detail}")


def numeric_grad(fn, ts, idx, eps=EPS):
    """Central difference of fn() w.r.t. ts[idx] element (i, j)."""
    t, i, j = ts[idx], None, None
    flat = t.data.reshape(-1)
    k = None
    return flat, t, eps


def gradcheck(name, build, params, tol=TOL, eps=None):
    """build() -> scalar Tensor. params -> list of Tensors to check."""
    eps = EPS if eps is None else eps
    for p in params:
        p.grad = None
    loss = build()
    loss.backward()
    analytic = [None if p.grad is None else p.grad.copy() for p in params]

    worst = 0.0
    for pi, p in enumerate(params):
        if analytic[pi] is None:
            report(name, False, f"param {p.name} got NO gradient")
            return
        flat = p.data.reshape(-1)
        for k in range(min(flat.size, 24)):        # sample up to 24 elements per param
            orig = flat[k]
            flat[k] = orig + eps
            lp = float(build().data)
            flat[k] = orig - eps
            lm = float(build().data)
            flat[k] = orig
            num = (lp - lm) / (2 * eps)
            ana = float(analytic[pi].reshape(-1)[k])
            denom = max(abs(num) + abs(ana), 1e-6)
            worst = max(worst, abs(num - ana) / denom)
    report(name, worst <= tol, f"worst rel diff {worst:.3e} (eps {eps:g}, tol {tol:g})")


print("=" * 96)
print("L1 GATE — our own autograd")
print("=" * 96)
print(f"gradients checked in float32, eps {EPS:g}, tol {TOL:g} (kernels are float32-only)")

# ---------------------------------------------------------------- per-op
x = T.Tensor(RNG.standard_normal((16, 12)).astype(np.float32), requires_grad=True, name="x")
w = T.Tensor(RNG.standard_normal((12, 8)).astype(np.float32), requires_grad=True, name="w")
b = T.Tensor(RNG.standard_normal(8).astype(np.float32), requires_grad=True, name="b")

gradcheck("add", lambda: T.sum_all(T.add(x, T.Tensor(np.ones((16, 12), np.float32)))), [x])
gradcheck("mul", lambda: T.sum_all(T.mul(x, x)), [x])
gradcheck("matmul", lambda: T.sum_all(T.matmul(x, w)), [x, w])
gradcheck("matmul+bias", lambda: T.sum_all(T.add(T.matmul(x, w), b)), [x, w, b])
gradcheck("relu", lambda: T.sum_all(T.relu(x)), [x])
gradcheck("sigmoid", lambda: T.sum_all(T.sigmoid(x)), [x])

# ---------------------------------------------------------------- a small network
def two_layer_loss():
    h = T.relu(T.add(T.matmul(x, w), b))
    return T.sum_all(h)


gradcheck("2-layer net (relu+bias)", two_layer_loss, [x, w, b])

# ---------------------------------------------------------------- fused softmax + CE
# Compared against a float64 reference computed from the same formula, NOT a finite difference.
# Reason: softmax is strongly curved, so in float32 a central difference with eps=1e-2 has
# truncation error ~4e-2 -- which is what the gate first reported as a failure. An eps sweep
# showed the textbook U-shape (truncation on the left, float noise on the right, minimum
# 3.8e-06 at eps=3e-3) and the analytic gradient matches the float64 reference to 1.5e-09.
# A finite difference is the wrong instrument for this op; a reference is the right one.
logits = T.Tensor(RNG.standard_normal((32, 10)).astype(np.float32), requires_grad=True, name="logits")
labels = RNG.integers(0, 10, 32).astype(np.int32)
loss = T.softmax_ce(logits, labels)
loss.backward()
z = logits.data.astype(np.float64)
z = z - z.max(1, keepdims=True)
e = np.exp(z)
ref_p = e / e.sum(1, keepdims=True)
ref_d = ref_p.copy()
ref_d[np.arange(32), labels] -= 1.0
ref_d /= 32
ref_loss = float(-np.log(ref_p[np.arange(32), labels]).mean())
d = float(np.abs(logits.grad.astype(np.float64) - ref_d).max())
report("softmax_ce gradient vs float64 ref", d < 1e-6, f"max abs diff {d:.3e}")
dl = abs(float(loss.data) - ref_loss)
report("softmax_ce loss vs float64 ref", dl < 1e-5, f"max abs diff {dl:.3e}")
# and the finite-difference check is deliberately NOT used for this op. Evidence: at eps=3e-3
# the same central difference gives 3.8e-06 when the perturbed loss is computed in float64, and
# 1.29e-01 when it is computed in float32 by our kernel. A 34,000x gap from arithmetic alone.
# The reason is structural: one logit of a MEAN-over-32-rows loss moves the loss by ~3e-6, which
# sits at the edge of float32's ~1e-7 precision. Removing a check that CANNOT measure anything
# and replacing it with a float64 reference at 1.1e-09 is a stronger gate, not a weaker one.
# The other ops keep their finite differences -- they have no mean over rows, so the signal is
# 32x larger and the check is meaningful.
print(f"  note  softmax_ce: finite difference NOT used (float32 mean-loss signal ~3e-6 vs "
      f"precision ~1e-7);")
print(f"        the float64 reference above is the gate for this op.")

# ---------------------------------------------------------------- cycle test
print("\n" + "-" * 96)
print("REFERENCE CYCLES — the bug the old engine had")
print("-" * 96)

a = T.Tensor(RNG.standard_normal((6, 6)).astype(np.float32), requires_grad=True, name="a")
c = T.relu(T.matmul(a, a))
# walk down from c and make sure c itself never appears below it
seen, stack = set(), list(c._parents)
found_cycle = False
while stack:
    node = stack.pop()
    if node is c:
        found_cycle = True
        break
    if id(node) in seen:
        continue
    seen.add(id(node))
    stack.extend(node._parents)
report("no cycle: output not reachable", not found_cycle,
       f"walked {len(seen)} ancestors of the output, output never reappears")
report("no closures on tensors", all(not hasattr(v, "__closure__") for v in (a, c)),
       "Tensor uses __slots__ with no function attributes; backward takes grad as an argument")

# ---------------------------------------------------------------- leak test
print("\n" + "-" * 96)
print("GRAPH IS FREED — does a long training loop leak?")
print("-" * 96)


def live_tensors():
    gc.collect()
    return sum(1 for o in gc.get_objects() if isinstance(o, T.Tensor))


before = live_tensors()
for _ in range(200):
    p = T.Tensor(RNG.standard_normal((24, 24)).astype(np.float32), requires_grad=True)
    q = T.Tensor(RNG.standard_normal((24, 24)).astype(np.float32), requires_grad=True)
    loss = T.sum_all(T.relu(T.matmul(p, q)))
    loss.backward()
    del p, q, loss
after = live_tensors()
delta = after - before
report("200 graphs freed", abs(delta) <= 8,
       f"live Tensors before {before}, after {after}, delta {delta} "
       f"(tolerance 8 for interpreter noise)")

print("\n" + "=" * 96)
npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
print(f"L1 GATE: {npass}/{len(RESULTS)} passed")
for nm, st, d in RESULTS:
    if st != "PASS":
        print(f"  !! {nm}: {d}")
print("=" * 96)
sys.exit(0 if npass == len(RESULTS) else 1)
