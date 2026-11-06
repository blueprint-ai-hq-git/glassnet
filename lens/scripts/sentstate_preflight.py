"""Pre-flight for lens/sentstate.py (predictions P111a-h, lens/runs/SENTSTATE-PREDICTIONS-P111.txt).

CPU ONLY, NO CORPUS, NO GPU, seconds. Touches nothing that exists: lens/model.py and
lens/scripts/train_chat.py are only IMPORTED (and only to count parameters), never edited, so the
agreement arm that is training right now (agree_sgd04_s2) keeps the code it loaded.

Run:  python3 lens/scripts/sentstate_preflight.py 2>&1 | tee lens/runs/SENTSTATE-STRUCTURAL.txt
Exit code 0 = every predicted bar met, 1 = at least one FAIL (and a FAIL is printed as a FAIL).
"""
import os
import sys

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, P)

from lens import backend as BE                                    # noqa: E402
from lens import tensor as T                                      # noqa: E402
from lens.concepts import Concept                                 # noqa: E402
from lens.model import Dense, LensNet                             # noqa: E402
from lens.sentstate import SentenceState                          # noqa: E402

T.set_device("cpu")
T.set_dtype(np.float64)          # the finite-difference control needs headroom; float32 FD is noise

K, DS, V, D, W = 36, 64, 12002, 256, 32
fails = []


def verdict(pid, ok, detail):
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        fails.append(pid)


print("=" * 100)
print("SENTSTATE PRE-FLIGHT — P111a-h  (lens/sentstate.py; predictions written before this file existed)")
print("=" * 100)

# ------------------------------------------------------------------ P111a build + param arithmetic
ss = SentenceState(K, DS, rng=np.random.default_rng(0))
n_state = sum(int(np.asarray(p.data).size) for p in ss.parameters())
arith = K * DS + DS * DS + DS
per = {p.name: int(np.asarray(p.data).size) for p in ss.parameters()}
verdict("P111a", n_state == arith,
        f"params={n_state:,} (Wc {per['sstate.Wc']:,} + Ur {per['sstate.Ur']:,} + b {per['sstate.b']:,}) "
        f"vs arithmetic {arith:,} -> {'exact' if n_state == arith else 'MISMATCH'}")

# ------------------------------------------------------------------ P111b determinism
rng = np.random.default_rng(1)
h = T.Tensor(rng.standard_normal((256, K)).astype(np.float32), requires_grad=False, name="h")
s0 = ss.initial(256)
sa = ss(h, s0)
sb = ss(h, s0)
d_det = float(np.max(np.abs(np.asarray(BE.to_cpu(sa.data)) - np.asarray(BE.to_cpu(sb.data)))))
verdict("P111b", d_det == 0.0, f"max|A-B| = {d_det:.1e} (bar: exactly 0.0)")

# ------------------------------------------------------------------ P111c no-flag control on LensNet
cs = [Concept(f"c{k}", [3 + k, 40 + k, 90 + k]) for k in range(K)]
twin = LensNet(V, cs, d=D, window=W, seed=0, lam_concept=0.3, lam_act=0.0,
               direct=True, direct_mode="tied")
n_twin = sum(int(np.asarray(p.data).size) for p in twin.parameters())
ctx = np.ascontiguousarray(rng.integers(1, V, size=(2, W)).astype(np.int64))
lg = twin.forward(ctx)["logits"]
finite = bool(np.isfinite(np.asarray(BE.to_cpu(lg.data))).all())
verdict("P111c", n_twin == 5908686 and finite,
        f"LensNet(d=256,W=32,K=36,tied) params={n_twin:,} (stored H2L count 5,908,686), "
        f"forward finite={finite} -> model.py untouched this turn: "
        f"{'YES' if n_twin == 5908686 else 'NO — INVESTIGATE BEFORE ANY ARM'}")

# ------------------------------------------------------------------ P111d the state is READ
cfg = np.random.default_rng(2)
h_fix = T.Tensor(cfg.standard_normal((8, K)), requires_grad=False, name="h_fix")
s_zero = ss.initial(8)
s_half = np.full((8, DS), 0.5, dtype=np.float32)
d_prev = float(np.max(np.abs(np.asarray(BE.to_cpu(ss(h_fix, s_zero).data))
                             - np.asarray(BE.to_cpu(ss(h_fix, s_half).data)))))
h_zero = T.Tensor(np.zeros((8, K), dtype=np.float32), requires_grad=False, name="h_zero")
h_ones = T.Tensor(np.ones((8, K), dtype=np.float32), requires_grad=False, name="h_ones")
d_h = float(np.max(np.abs(np.asarray(BE.to_cpu(ss(h_zero, s_zero).data))
                          - np.asarray(BE.to_cpu(ss(h_ones, s_zero).data)))))
verdict("P111d", d_prev > 1e-3 and d_h > 1e-3,
        f"max|s(s_prev=0) - s(s_prev=0.5)| = {d_prev:.4f}; max|s(h=0) - s(h=1)| = {d_h:.4f} "
        f"(bar: both > 1e-3)")

# ------------------------------------------------------------------ P111e bounded, shrinking, not collapsing
# INSTRUMENT FIX (attempt 1 printed PASS-able numbers but at 4 decimals; and the stress input was N(0,1),
# which is NOT what the concept layer produces — `h = sigmoid(...)` lives in (0, 1)). Both cases are run and
# reported; the SHRINK clause is the one the attempt-1 FAIL was about and it is unchanged.
st = ss.initial(32)
worst_norm = 0.0
for step in range(64):
    hh = T.Tensor(np.random.default_rng(100 + step).standard_normal((32, K)).astype(np.float32),
                  requires_grad=False)
    st = np.asarray(BE.to_cpu(ss(hh, st).data))
    worst_norm = max(worst_norm, float(np.max(np.abs(st))))
st_r = ss.initial(32)
worst_real = 0.0
for step in range(64):
    hr = T.Tensor(np.random.default_rng(500 + step).random((32, K)).astype(np.float32),
                  requires_grad=False)
    st_r = np.asarray(BE.to_cpu(ss(hr, st_r).data))
    worst_real = max(worst_real, float(np.max(np.abs(st_r))))
silent = np.full((32, DS), 0.9, dtype=np.float32)
zero_h = T.Tensor(np.zeros((32, K), dtype=np.float32), requires_grad=False)
shrink = float(np.max(np.abs(np.asarray(BE.to_cpu(ss(zero_h, silent).data)))))
s_next = np.asarray(BE.to_cpu(ss(zero_h, np.asarray(BE.to_cpu(ss(zero_h, silent).data))).data))
not_collapsed = float(np.max(np.abs(s_next)))
verdict("P111e", worst_norm <= 1.0 + 1e-9 and shrink < 0.9 and not_collapsed > 1e-4,
        f"64-step max|s| = {worst_norm:.12f} under N(0,1) inputs and {worst_real:.6f} under (0,1) "
        f"concept-like inputs (bar <= 1.0); silent step from 0.9 -> {shrink:.6f} (bar < 0.9); next "
        f"silent step -> {not_collapsed:.6f} (bar > 1e-4)")

# ------------------------------------------------------------------ P111f gradients + finite differences
# INSTRUMENT FIX (attempt 1 reported FAIL "all-nonzero=False" and it was MY test, not the module): the
# gradient test passed s_prev = 0, and with a zero previous state d(state)/d(Ur) is EXACTLY zero by the
# chain rule — the analytic and numeric gradients both read 0.00000000, which is correct maths. The carry
# is therefore tested here with a NON-ZERO previous state, and the zero-prev case is kept as a CONTROL
# that must read exactly 0 (BUGS.md [BUG-SENTSTATE-PREFLIGHT-ZERO-PREV]).
ss2 = SentenceState(K, DS, rng=np.random.default_rng(7))
h2 = T.Tensor(np.random.default_rng(3).standard_normal((6, K)), requires_grad=False)
s_prev2 = np.full((6, DS), 0.5, dtype=np.float32)


def loss_value(s_prev=s_prev2):
    return float(np.asarray(BE.to_cpu(T.sum_all(ss2(h2, s_prev)).data)))


out = ss2(h2, s_prev2)
T.sum_all(out).backward()
grads = {p.name: np.asarray(BE.to_cpu(p.grad)) if p.grad is not None else None
         for p in ss2.parameters()}
all_finite = all(g is not None and np.isfinite(g).all() for g in grads.values())
nonzero = all(g is not None and np.any(g != 0.0) for g in grads.values())

# CONTROL: a zero previous state must give an exactly-zero Ur gradient (correct maths, not a bug)
ss3 = SentenceState(K, DS, rng=np.random.default_rng(7))
T.sum_all(ss3(h2, np.zeros((6, DS), dtype=np.float32))).backward()
g_ur_zero = float(np.max(np.abs(np.asarray(BE.to_cpu(ss3.Ur.grad)))))
g_wc_zero = float(np.max(np.abs(np.asarray(BE.to_cpu(ss3.Wc.grad)))))
zero_prev_control = (g_ur_zero == 0.0) and (g_wc_zero > 0.0)

eps = 1e-6
fd_errs = []
for tag, par, idxs in (("Wc", ss2.Wc, [(0, 0), (K // 2, DS // 3), (K - 1, DS - 1)]),
                       ("Ur", ss2.Ur, [(0, 0), (DS // 2, DS // 3), (DS - 1, DS - 1)])):
    Wd = np.asarray(BE.to_cpu(par.data)).copy()
    g_an = np.asarray(BE.to_cpu(par.grad))
    for (i, j) in idxs:
        base = Wd[i, j]
        Wd[i, j] = base + eps
        par.data = BE.to_dev(Wd)
        lp = loss_value()
        Wd[i, j] = base - eps
        par.data = BE.to_dev(Wd)
        lm = loss_value()
        Wd[i, j] = base
        par.data = BE.to_dev(Wd)
        num = (lp - lm) / (2 * eps)
        ana = float(g_an[i, j])
        rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
        fd_errs.append((f"{tag}[{i},{j}]", ana, num, rel))
worst_rel = max(e[3] for e in fd_errs)
for tag, ana, num, rel in fd_errs:
    print(f"      fd {tag:>12}: analytic {ana: .8f}  numeric {num: .8f}  rel {rel:.2e}")
verdict("P111f", all_finite and nonzero and worst_rel < 1e-3 and zero_prev_control,
        f"grads finite={all_finite} all-nonzero={nonzero}; worst finite-difference rel err "
        f"{worst_rel:.2e} over 6 entries (bar < 1e-3); zero-prev control: max|dUr| = {g_ur_zero:.1e} "
        f"(must be exactly 0) max|dWc| = {g_wc_zero:.2e} (must be > 0) -> {zero_prev_control}")

# ------------------------------------------------------------------ P111g carry is truncated by construction
refused, msg = False, ""
try:
    ss(h_fix, T.Tensor(np.zeros((8, DS)), requires_grad=True))
except ValueError as e:
    refused, msg = True, str(e).split(".")[0]
legal = ss(h_fix, s_zero).data.shape
verdict("P111g", refused and tuple(legal) == (8, DS),
        f"requires_grad s_prev refused={refused} (\"{msg[:70]}...\"); legal path shape={tuple(legal)}")

# ------------------------------------------------------------------ P111h cost from the trainer's constants
ans_off = Dense(K, V, np.random.default_rng(11), name="answer_off")
ans_on = Dense(K + DS, V, np.random.default_rng(11), name="answer_on")
n_off = sum(int(np.asarray(p.data).size) for p in ans_off.parameters())
n_on = sum(int(np.asarray(p.data).size) for p in ans_on.parameters())
delta_ans = n_on - n_off
total = n_twin + n_state + delta_ans
pct = 100.0 * (n_state + delta_ans) / n_twin
verdict("P111h", delta_ans == V * DS and abs(pct - 13.11) < 0.02 and total < 50_000_000,
        f"answer {K}->{K + DS} inputs: {n_off:,} -> {n_on:,} (+{delta_ans:,} = V*ds "
        f"{V * DS:,}); state +{n_state:,}; total {total:,} (+{pct:.2f}% on 5,908,686, cap 50M)")

# ------------------------------------------------------------------ readability smoke (not a prediction)
print("-" * 100)
names = [f"c{k}" for k in range(K)]
hrow = np.zeros(K, dtype=np.float32)
hrow[[4, 17, 25]] = [0.9, 0.6, 0.3]
print("readability smoke — attribution of a state written by 3 named concepts (0.9/0.6/0.3 at k=4/17/25):")
for nm, hv, mag in ss.explain_state(hrow, np.zeros(DS, dtype=np.float32), names, top=3):
    print(f"      {nm:>5} value {hv: .3f}  ||contribution|| {mag:.4f}")
print("-" * 100)
print(f"RESULT: {len(fails)} FAIL(s){'' if not fails else ' -> ' + ', '.join(fails)}")
sys.exit(0 if not fails else 1)
