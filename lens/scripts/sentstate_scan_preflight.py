"""Pre-flight for the position scan (predictions P112a-f, lens/runs/SENTSTATE-SCAN-PREDICTIONS-P112.txt).

CPU ONLY, NO CORPUS, NO GPU. Imports lens/sentstate.py; edits nothing that exists.

Run:  bash lens/scripts/run_sentstate_scan_preflight.sh "<reason>"
Exit 0 = every predicted bar met; 1 = at least one FAIL, printed as FAIL.
"""
import os
import sys

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, P)

from lens import backend as BE                                    # noqa: E402
from lens import tensor as T                                      # noqa: E402
from lens.sentstate import PosConcepts, SentenceState, carry_scan, sstate_scan   # noqa: E402

T.set_device("cpu")
T.set_dtype(np.float64)

fails = []


def verdict(pid, ok, detail):
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        fails.append(pid)


def ref_loop(hp, Wc, Ur, b):
    """Plain Python reference: the recurrence spelled out, no autograd, no engine."""
    n, wt, _k = hp.shape
    s = np.zeros((n, Ur.shape[0]))
    for t in range(wt):
        s = np.tanh(hp[:, t, :] @ Wc + s @ Ur + b)
    return s


print("=" * 100)
print("SENTSTATE SCAN PRE-FLIGHT — P112a-f  (predictions written before lens/sentstate.py gained a scan)")
print("=" * 100)

N, WT, K, DS = 4, 8, 5, 6
cfg = np.random.default_rng(21)
hp_np = cfg.random((N, WT, K))                     # concept vectors live in (0,1): sigmoid outputs
st = SentenceState(K, DS, rng=np.random.default_rng(3))
Wc = np.asarray(BE.to_cpu(st.Wc.data))
Ur = np.asarray(BE.to_cpu(st.Ur.data))
b = np.asarray(BE.to_cpu(st.b.data))

# ------------------------------------------------------------------ P112a forward == the recurrence
hp3 = T.Tensor(hp_np, requires_grad=False, name="hp3")
scan_out = np.asarray(BE.to_cpu(sstate_scan(hp3, st.Wc, st.Ur, st.b).data))
ref = ref_loop(hp_np, Wc, Ur, b)
d_ref = float(np.max(np.abs(scan_out - ref)))
verdict("P112a", d_ref == 0.0, f"max|scan - reference loop| = {d_ref:.1e} (bar: exactly 0.0)")

# ------------------------------------------------------------------ P112d W=1 == the P111 single step
hp1 = T.Tensor(hp_np[:, :1, :], requires_grad=False, name="hp1")
one_scan = np.asarray(BE.to_cpu(sstate_scan(hp1, st.Wc, st.Ur, st.b).data))
one_call = np.asarray(BE.to_cpu(st(T.Tensor(hp_np[:, 0, :], requires_grad=False), st.initial(N)).data))
d_one = float(np.max(np.abs(one_scan - one_call)))
verdict("P112d", d_one == 0.0, f"max|scan(W=1) - SentenceState.__call__| = {d_one:.1e} (bar: exactly 0.0)")

# ------------------------------------------------------------------ P112e bounded + order-sensitive
worst = float(np.max(np.abs(scan_out)))
perm = np.random.default_rng(9).permutation(WT)
hp_perm = hp_np[:, perm, :]
perm_out = np.asarray(BE.to_cpu(sstate_scan(T.Tensor(hp_perm, requires_grad=False),
                                            st.Wc, st.Ur, st.b).data))
d_order = float(np.max(np.abs(perm_out - scan_out)))
verdict("P112e", worst <= 1.0 + 1e-12 and d_order > 1e-3,
        f"max|s| = {worst:.12f} (bar <= 1); same multiset reordered -> max|ds| = {d_order:.4f} "
        f"(bar > 1e-3, i.e. the state encodes ORDER)")

# ------------------------------------------------------------------ P112b gradients + finite differences
# the full chain the trainer would use: per-token representation -> per-token concepts -> scan -> answer
D = 7
pc = PosConcepts(D, K, rng=np.random.default_rng(5))
x_rows = T.Tensor(np.random.default_rng(6).random((N * WT, D)), requires_grad=False, name="x")
hp_rows = pc(x_rows)
s_term = carry_scan(st, hp_rows, N, WT)
loss = T.sum_all(s_term)
loss.backward()

grads = {p.name: (None if p.grad is None else np.asarray(BE.to_cpu(p.grad)))
         for p in st.parameters() + pc.parameters()}
all_finite = all(g is not None and np.isfinite(g).all() for g in grads.values())
nonzero = all(g is not None and np.any(g != 0.0) for g in grads.values())


def loss_val():
    return float(np.asarray(BE.to_cpu(T.sum_all(carry_scan(st, pc(x_rows), N, WT)).data)))


eps = 1e-6
fd = []
checks = (("Wc", st.Wc, [(0, 0), (K // 2, DS // 3), (K - 1, DS - 1)]),
          ("Ur", st.Ur, [(0, 0), (DS // 2, DS // 3), (DS - 1, DS - 1)]),
          ("b", st.b, [0, DS // 2, DS - 1]),
          ("Wc2", pc.Wc2, [(0, 0), (D // 2, K // 2), (D - 1, K - 1)]))
for tag, par, idxs in checks:
    Wd = np.asarray(BE.to_cpu(par.data)).copy()
    g_an = np.asarray(BE.to_cpu(par.grad))
    for idx in idxs:
        idx = idx if isinstance(idx, tuple) else (idx,)
        base = Wd[idx]
        Wd[idx] = base + eps
        par.data = BE.to_dev(Wd)
        lp = loss_val()
        Wd[idx] = base - eps
        par.data = BE.to_dev(Wd)
        lm = loss_val()
        Wd[idx] = base
        par.data = BE.to_dev(Wd)
        num = (lp - lm) / (2 * eps)
        ana = float(g_an[idx])
        rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
        fd.append((f"{tag}{list(idx)}", ana, num, rel))
worst_rel = max(e[3] for e in fd)
for tag, ana, num, rel in fd:
    print(f"      fd {tag:>14}: analytic {ana: .9f}  numeric {num: .9f}  rel {rel:.2e}")
verdict("P112b", all_finite and nonzero and worst_rel < 1e-3,
        f"grads finite={all_finite} all-nonzero={nonzero}; worst finite-difference rel err "
        f"{worst_rel:.2e} over {len(fd)} entries across Wc/Ur/b/Wc2 (bar < 1e-3)")

# ------------------------------------------------------------------ P112c the per-position gradient decays with distance
hp_g = T.Tensor(hp_np, requires_grad=True, name="hp_g")
T.sum_all(sstate_scan(hp_g, st.Wc, st.Ur, st.b)).backward()
gh = np.asarray(BE.to_cpu(hp_g.grad))
g_first = float(np.max(np.abs(gh[:, 0, :])))
g_last = float(np.max(np.abs(gh[:, WT - 1, :])))
g_mid = float(np.max(np.abs(gh[:, WT // 2, :])))
verdict("P112c", g_first > 0 and g_last > 0 and g_first < g_last,
        f"max|dL/dhp| at t=0 {g_first:.5f}, t={WT // 2} {g_mid:.5f}, t={WT - 1} {g_last:.5f} "
        f"(bar: both ends nonzero and t=0 < t=last -> {g_first < g_last})")

# ------------------------------------------------------------------ P112f cost from the trainer's constants
D_ship, K_ship, DS_ship, V_ship, base = 256, 36, 64, 12002, 5908686
wc2 = D_ship * K_ship
state = K_ship * DS_ship + DS_ship * DS_ship + DS_ship
ans = V_ship * DS_ship
added = wc2 + state + ans
pct = 100.0 * added / base
verdict("P112f", wc2 == 9216 and state == 6464 and ans == 768128 and abs(pct - 13.27) < 0.02,
        f"Wc2 {wc2:,} + state {state:,} + answer delta {ans:,} = {added:,} = +{pct:.2f}% on {base:,} "
        f"(total {base + added:,}, cap 50M)")

print("-" * 100)
print(f"RESULT: {len(fails)} FAIL(s){'' if not fails else ' -> ' + ', '.join(fails)}")
sys.exit(0 if not fails else 1)
