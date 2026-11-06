"""run_distill_real.py — CAN OUR BEST MODEL MAKE A SMALLER MODEL SMARTER? (real corpus)

WHY THIS EXISTS
---------------
The first distillation study (run_distill.py) ran on tiny_data: vocab 1500, a weak teacher
(acc 29.6%), 3 seeds -> student A ahead by +0.125 pts, INSIDE seed noise. The post-mortem
was already written: the teacher was too weak to teach, and the corpus too small to matter.
This run fixes BOTH:

    TEACHER   the SHIPPED talk line (t11_talk41_ep192: d256, W32, v12000, 192 epochs,
              30.88% val acc) -- the best model this project owns, no training needed
    STUDENT-A d128, same real corpus, real next-token loss PLUS the teacher's soft
              distribution (KL), lam_soft from DR_LAM_SOFT
    STUDENT-B d128, identical in every way EXCEPT it never sees the teacher (the control)

Same vocab id-space (both built from the same v12000 cache), same windows, same order, same
seed, same steps. The ONLY difference is the soft term.

PRE-REGISTERED PREDICTIONS (written BEFORE running -- project rule)
    P1  the teacher scores 30-32% on the PLAIN val pool (its ship log says val_acc
        30.94%, val_ppl 38311 -- plain-pool semantics, the model never learned to
        predict human tokens). MEASURED during the smoke: plain 38.00% on the first
        4000 val windows, and ASSISTANT-MASKED ppl 11.03 / acc 80.59% on its own
        words -- the masked number is the one a chatbot should be judged on, and it
        is the comparator the students face below.
    P2  both students land BELOW the teacher (d128 vs d256) -- expected
    P3  student A beats student B by MORE THAN 2 points, with the per-seed deltas all
        positive (outside seed noise) -> the teacher's soft information transfers
    P4  if P3 fails, the honest conclusion is that on this corpus a same-family teacher
        carries no information a same-data student cannot get itself -- a MAPPED NEGATIVE
        at this rung, reported as NO, and the distillation line closes on evidence here.

Usage: bash lens/scripts/run_distill_real.sh   (env knobs: DR_EPOCHS, DR_SEEDS, DR_WINDOWS,
DR_LAM_SOFT, DR_TEACHER, DR_D_STUDENT)
"""
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import chat_concepts_wide                                   # noqa: E402
import chat_data                                            # noqa: E402
import lens                                                 # noqa: E402
import lens.tensor as T                                     # noqa: E402
from lens import backend as B                               # noqa: E402
from lens.concepts import Concept                           # noqa: E402

WIN = 32
EPOCHS = int(os.environ.get("DR_EPOCHS", "8"))
BATCH, LR = 256, 0.4
SEEDS = [int(s) for s in os.environ.get("DR_SEEDS", "0,1,2").split(",")]
LAM_SOFT = float(os.environ.get("DR_LAM_SOFT", "0.5"))
MAXW = int(os.environ.get("DR_WINDOWS", "400000"))
D_STUDENT = int(os.environ.get("DR_D_STUDENT", "128"))
TEACHER_CKPT = os.environ.get("DR_TEACHER", "lens/runs/chat/t11_talk41_ep192.npz")
DEV = os.environ.get("DR_DEVICE", "gpu")
OUT = os.path.join(PROJ, "lens", "runs", "distill_real")
os.makedirs(OUT, exist_ok=True)


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- the soft-KL backward
def _bw_soft_kl(out, parents, saved, grad):
    """d/dlogits of -sum_t p_t log softmax(logits) = softmax(logits) - p_t, averaged."""
    lg = parents[0]
    if not lg.requires_grad:
        return
    x = lg.data
    e = B.xp.exp(x - x.max(axis=1, keepdims=True))
    p_s = e / e.sum(axis=1, keepdims=True)
    d = (p_s - saved["p_t"]) / saved["n"]
    T._accumulate(lg, d * T._scalar(grad))


T.OPS["soft_kl"] = _bw_soft_kl

log("=" * 100)
log("REAL-CORPUS DISTILLATION — our best model teaching a smaller one")
log("=" * 100)

# ---------------------------------------------------------------- data (the talk line's own)
d = chat_data.build()
ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
Xtr, Ytr = chat_data.make_windows(ids, bounds, split, WIN, which=0)
Xva, Yva = chat_data.make_windows(ids, bounds, split, WIN, which=1)
if len(Xtr) > MAXW:
    Xtr, Ytr = Xtr[:MAXW], Ytr[:MAXW]
train_mask = np.zeros(len(ids), dtype=bool)
for j in range(len(bounds) - 1):
    if split[j] == 0:
        train_mask[bounds[j]:bounds[j + 1]] = True
cs, spec = chat_concepts_wide.build(vocab, ids, train_mask, windows=Xtr)
concepts = [Concept(c["name"], c["ids"], support=c["support"]) for c in cs]
log(f"data: {len(Xtr):,} train / {len(Xva):,} val windows, vocab {len(vocab):,}, "
    f"{len(concepts)} concepts, device {DEV}")

# --- assistant-only mask: THE TALK LINE'S OWN OBJECTIVE -------------------------------------
# The shipped teacher was trained with human targets weighted 0. A student trained on a
# different objective would not be comparable, and the teacher's own numbers only reproduce
# under the mask (plain-pool ppl explodes on human tokens it never learned to predict).
is_asst = np.zeros(len(ids), dtype=np.float32)
in_asst = False
for j in range(len(bounds) - 1):
    a0, b0 = bounds[j], bounds[j + 1]
    toks = [vocab[i] for i in ids[a0:b0]]
    k = 0
    while k < len(toks):
        if toks[k].lower() == "assistant" and k + 1 < len(toks) and toks[k + 1] == ":":
            in_asst = True
            k += 2
            continue
        if toks[k].lower() == "human" and k + 1 < len(toks) and toks[k + 1] == ":":
            in_asst = False
            k += 2
            continue
        if in_asst:
            is_asst[a0 + k] = 1.0
        k += 1
Wtr, Wva = is_asst[Ytr], is_asst[Yva]
log(f"assistant-only: {float(Wtr.mean())*100:.1f}% of train targets, "
    f"{float(Wva.mean())*100:.1f}% of val targets")

# ---------------------------------------------------------------- the teacher (already trained)
tpath = os.path.join(PROJ, TEACHER_CKPT)
if not os.path.exists(tpath):
    log(f"TEACHER MISSING: {tpath} -> NO VERDICT")
    json.dump({"verdict": "NO VERDICT", "reason": "teacher checkpoint missing"},
              open(os.path.join(OUT, "verdict.json"), "w"), indent=1)
    sys.exit(0)
teacher = lens.load(tpath, device=DEV)
tV, tW = getattr(teacher, "V", None), getattr(teacher, "W", None)
if tV is not None and int(tV) != len(vocab):
    log(f"VOCAB DRIFT: teacher V={tV} vs data V={len(vocab)} -- refusing to distil across "
        f"id spaces")
    sys.exit(2)
if tW is not None and int(tW) != WIN:
    log(f"WINDOW DRIFT: teacher W={tW} vs data W={WIN} -- refusing")
    sys.exit(2)
log(f"teacher: {os.path.basename(tpath)}  d={getattr(teacher, 'D', '?')}  V={tV}  W={tW}  "
    f"params {sum(p.data.size for p in teacher.parameters()):,}")


def softmax_np(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def evaluate(net, X, y, n=4000, w=None):
    """Returns (ppl, acc) on the ASSISTANT-MASKED targets when w is given (comparable to the
    ship's own 30.88%/ppl), else the plain pool."""
    idx = np.arange(min(n, len(X)))
    lg = net.forward_arrays(X[idx])["logits"]
    p = softmax_np(lg)
    if w is not None:
        m = np.asarray(w)[idx] > 0.5
        if m.sum() == 0:
            return float("nan"), float("nan")
        p = p[m]
        tgt = y[idx][m]
        ii = np.arange(len(tgt))
    else:
        tgt = y[idx]
        ii = np.arange(len(idx))
    ppl = float(np.exp(-np.log(np.maximum(p[ii, tgt], 1e-12)).mean()))
    acc = float((p.argmax(axis=1) == tgt).mean())
    return ppl, acc


tp, ta = evaluate(teacher, Xva, Yva, w=Wva)
tp_p, ta_p = evaluate(teacher, Xva, Yva)
log(f"TEACHER held-out (assistant-masked): ppl {tp:.3f}  acc {ta*100:.2f}%   "
    f"(P1 predicted 30-32%; plain pool ppl {tp_p:.1f} acc {ta_p*100:.2f}%)")


def build(d_, seed):
    return lens.Net(vocab=len(vocab), concepts=concepts, d=d_, window=WIN, seed=seed,
                    lam_concept=0.3, lam_act=0.2, device=DEV)


def train_student(net, seed, epochs, teacher=None, lam_soft=0.0, tag=""):
    """Real next-token loss (+ the teacher's soft distribution when teacher is not None).
    Soft targets are computed PER BATCH -- the full matrix would be N x V."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(Xtr))
        tot, nb = 0.0, 0
        for s in range(0, len(Xtr), BATCH):
            idx = order[s:s + BATCH]
            for p in net.parameters():
                p.grad = None
            loss, _parts = net.loss(Xtr[idx], Ytr[idx], weights=Wtr[idx])
            if teacher is not None and lam_soft > 0:
                lg_t = teacher.forward(Xtr[idx])["logits"]
                p_t = B.to_dev(softmax_np(B.to_cpu(lg_t.data)))
                lg = net.forward(Xtr[idx])["logits"]
                x = lg.data
                mx = x.max(axis=1, keepdims=True)
                logp = x - mx - B.xp.log(B.xp.exp(x - mx).sum(axis=1, keepdims=True))
                kl = -(p_t * logp).sum(axis=1).mean()
                soft_node = T.Tensor(kl, lg.requires_grad, _parents=(lg,), _op="soft_kl",
                                     _saved={"p_t": p_t, "n": len(idx)})
                loss = T.add(loss, T.mul(soft_node, T.Tensor(np.asarray(lam_soft, np.float32))))
            loss.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.data = p.data - LR * p.grad
            tot += float(B.to_cpu(loss.data))
            nb += 1
        if ep % 2 == 0 or ep == epochs:
            ppl, acc = evaluate(net, Xva, Yva, w=Wva)
            log(f"    [{tag}] ep {ep:>3}  loss {tot/max(nb,1):.4f}  val ppl {ppl:.3f}  "
                f"acc {acc*100:.2f}%  ({time.perf_counter()-t0:.0f}s)")
    return time.perf_counter() - t0


rows = []
for seed in SEEDS:
    log(f"\n--- seed {seed} ---")
    a = build(D_STUDENT, seed)
    train_student(a, seed, EPOCHS, teacher=teacher, lam_soft=LAM_SOFT, tag=f"A s{seed}")
    ap, aa = evaluate(a, Xva, Yva, w=Wva)
    b = build(D_STUDENT, seed)
    train_student(b, seed, EPOCHS, teacher=None, tag=f"B s{seed}")
    bp, ba = evaluate(b, Xva, Yva, w=Wva)
    log(f"  seed {seed}: A {aa*100:.2f}% / ppl {ap:.3f}   B {ba*100:.2f}% / ppl {bp:.3f}   "
        f"delta {100*(aa-ba):+.2f} pts")
    rows.append({"seed": seed, "a_acc": aa, "b_acc": ba, "a_ppl": ap, "b_ppl": bp,
                 "delta_acc": aa - ba, "delta_ppl": ap - bp})
    lens.save(a, os.path.join(OUT, f"student_a_s{seed}.npz"))
    lens.save(b, os.path.join(OUT, f"student_b_s{seed}.npz"))

da = np.array([r["delta_acc"] for r in rows]) * 100
dp = np.array([r["delta_ppl"] for r in rows])
log("\n" + "=" * 100)
log("RESULT")
log("=" * 100)
log(f"  teacher: {ta*100:.2f}%  ({sum(p.data.size for p in teacher.parameters()):,} params)")
for r in rows:
    log(f"  seed {r['seed']}: A {r['a_acc']*100:6.2f}%   B {r['b_acc']*100:6.2f}%   "
        f"delta {r['delta_acc']*100:+.2f} pts")
log(f"\n  MEAN delta {da.mean():+.3f} pts (std {da.std():.3f}, min {da.min():+.3f})   "
    f"ppl delta mean {dp.mean():+.3f} (lower is better)")
bar = 2.0
p3 = bool(da.mean() > bar and da.min() > 0)
log(f"  P3 (A beats B by > {bar} pts AND every seed positive): "
    f"{'PASSES' if p3 else 'FAILS'}")
log(f"  VERDICT: {'the teacher transfers information' if p3 else 'NO TRANSFER at this rung'}")
res = {"verdict": "transfer" if p3 else "no transfer", "teacher_acc": ta, "teacher_ppl": tp,
       "teacher_params": sum(p.data.size for p in teacher.parameters()),
       "lam_soft": LAM_SOFT, "epochs": EPOCHS, "windows": len(Xtr), "d_student": D_STUDENT,
       "rows": rows, "delta_acc_mean": float(da.mean()), "delta_acc_std": float(da.std()),
       "delta_acc_min": float(da.min()), "delta_ppl_mean": float(dp.mean()),
       "bar_pts": bar, "p3": p3}
json.dump(res, open(os.path.join(OUT, "verdict.json"), "w"), indent=1)
log(f"\nsaved {OUT}/verdict.json + per-seed checkpoints")
log("=" * 100)
