"""lens/scripts/run_distill.py — G3/G4: can a model make another model smarter, from scratch?

THE OWNER'S ASK (2026-10-07):
    "make two models while distillation size and one that is updated ... see which one performs the
     better the base model or the model that is trained from it but at the same size ... they should
     get smart if they didn't we have to continue"

So this trains THREE models and compares them:

    TEACHER    big (d=512), trained on the REAL corpus
    STUDENT-A  small (d=128), trained on the real corpus PLUS the teacher's soft predictions
    STUDENT-B  small (d=128), trained on the real corpus ONLY          <- the control

STUDENT-A and STUDENT-B have IDENTICAL architecture, identical size, identical data, identical
seed, identical steps. The ONLY difference is that A also sees what the teacher predicts. If the
teacher has anything worth transferring, A must beat B. If A does not beat B, the answer to the
owner's question is NO and it gets reported as NO.

WHY THE OLD ATTEMPT FAILED, AND WHAT CHANGED
--------------------------------------------
Target G (old engine) tested self-improvement and produced NO WIN. The post-mortem found the reason:
**the teacher was too weak to teach.** Its held-out accuracy was 2.00% against its own training
accuracy of 14-21%. A teacher that cannot do the task cannot teach the task, and the experiment was
measuring noise.

So TARGET-H.md added a precondition, and it is enforced here:

    **TEACHER PRECONDITION: if the teacher does not reach TEACHER_BAR on its own held-out set,
    this run reports NO VERDICT instead of a result.** A weak teacher cannot be used to claim
    anything about distillation, in either direction.

That bar is fixed BEFORE the run, at TEACHER_BAR = 25.0% next-token accuracy, which is close to what
a plain model of this size reaches on this corpus. It is not tuned afterwards.

WHAT "SOFT PREDICTIONS" MEANS
-----------------------------
For every training window the teacher produces a full probability distribution over the vocabulary.
The student is trained to match that distribution (cross-entropy against the teacher's softmax)
IN ADDITION to the real next token. This is standard distillation and it is computed entirely with
this project's own code -- the teacher is trained here, its outputs are generated here, and nothing
is downloaded.
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
import lens.tensor as T                                    # noqa: E402
from lens import backend as B                              # noqa: E402
from lens.concepts import Concept                          # noqa: E402
from glassnet import tiny_data                             # noqa: E402

WIN, VOCAB_CAP = 8, 1500
EPOCHS = int(os.environ.get("DS_EPOCHS", "40"))
BATCH, LR = 256, 0.5
SEED = int(os.environ.get("DS_SEED", "0"))
D_TEACHER, D_STUDENT = 512, 128
LAM_SOFT = float(os.environ.get("DS_LAM_SOFT", "0.5"))
TEACHER_BAR = 25.0                      # fixed BEFORE the run, per TARGET-H.md
OUT = os.path.join(PROJ, "lens", "runs", os.environ.get("DS_OUT", "distill"))
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("DS_DEVICE", "gpu")


def log(m):
    print(m, flush=True)


log("=" * 100)
log("DISTILLATION — can a model make a smaller model smarter? (teacher precondition enforced)")
log("=" * 100)

DATA = tiny_data.build(win=WIN, vocab_cap=VOCAB_CAP)
Xtr, ytr = np.asarray(DATA["Xtr"]), np.asarray(DATA["ytr"])
Xva, yva = np.asarray(DATA["Xva"]), np.asarray(DATA["yva"])
V = int(DATA["V"])
z = json.load(open(os.path.join(PROJ, "runs", "t3", "meaningful_targets_ext.json")))
concepts = [Concept(c["name"], c["ids"], support=float(c["support"])) for c in z["qualifying"]]
log(f"data {len(Xtr)} train / {len(Xva)} val, V={V}, {len(concepts)} concepts, device={DEV}")


def build(d, seed):
    return lens.Net(vocab=V, concepts=concepts, d=d, window=WIN, seed=seed,
                    lam_concept=0.3, lam_act=0.2, device=DEV)


def softmax_np(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def evaluate(net, X, y, n=4000):
    idx = np.arange(min(n, len(X)))
    lg = net.forward_arrays(X[idx])["logits"]
    p = softmax_np(lg)
    ppl = float(np.exp(-np.log(np.maximum(p[np.arange(len(idx)), y[idx]], 1e-12)).mean()))
    acc = float((p.argmax(axis=1) == y[idx]).mean())
    return ppl, acc


def train_plain(net, epochs, soft_targets=None, lam_soft=0.0, tag=""):
    """Train on the real next token. If soft_targets is given, ALSO match the teacher's distribution.

    The soft term is a cross-entropy against the teacher's probabilities rather than the one-hot
    label. Both terms are computed with this project's own ops.
    """
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(Xtr))
        tot, nb = 0.0, 0
        for s in range(0, len(Xtr), BATCH):
            idx = order[s:s + BATCH]
            for p in net.parameters():
                p.grad = None
            loss, parts = net.loss(Xtr[idx], ytr[idx])
            if soft_targets is not None and lam_soft > 0:
                out = net.forward(Xtr[idx])
                lg = out["logits"]
                tgt = soft_targets[idx]                       # (N, V) teacher probabilities
                logp = lg.data - B.xp.log(B.xp.exp(lg.data - lg.data.max(axis=1, keepdims=True))
                                          .sum(axis=1, keepdims=True)) - lg.data.max(axis=1,
                                                                                    keepdims=True)
                # KL(teacher || student) up to a constant: -sum_t p_t log p_s
                kl = -(B.to_dev(tgt) * logp).sum(axis=1).mean()
                soft_node = T.Tensor(kl, lg.requires_grad, _parents=(lg,), _op="soft_kl",
                                     _saved={"p_t": B.to_dev(tgt), "n": len(idx)})
                loss = T.add(loss, T.mul(soft_node,
                                         T.Tensor(np.asarray(lam_soft, np.float32))))
            loss.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.data = p.data - LR * p.grad
            tot += float(B.to_cpu(loss.data))
            nb += 1
        if ep % 10 == 0 or ep == epochs:
            ppl, acc = evaluate(net, Xva, yva)
            log(f"    [{tag}] ep {ep:>3}  loss {tot/max(nb,1):.4f}  val ppl {ppl:.3f}  "
                f"acc {acc*100:.2f}%")
    return time.perf_counter() - t0


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

# ---------------------------------------------------------------- 1. the teacher
log(f"\n--- TEACHER (d={D_TEACHER}) trained on the REAL corpus ---")
teacher = build(D_TEACHER, SEED)
t_t = train_plain(teacher, EPOCHS, tag="teacher")
tp, ta = evaluate(teacher, Xva, yva)
log(f"    teacher held-out: ppl {tp:.3f}   acc {ta*100:.2f}%")

if ta * 100 < TEACHER_BAR:
    log(f"\n  TEACHER PRECONDITION FAILED: teacher accuracy {ta*100:.2f}% < bar {TEACHER_BAR}%.")
    log(f"  A teacher that cannot do the task cannot teach the task. Per TARGET-H.md this run")
    log(f"  reports NO VERDICT rather than a result, in either direction.")
    with open(os.path.join(OUT, "verdict.json"), "w") as f:
        json.dump({"verdict": "NO VERDICT", "reason": "teacher below bar",
                   "teacher_acc": ta, "bar": TEACHER_BAR}, f, indent=1)
    sys.exit(0)

log(f"    teacher clears the {TEACHER_BAR}% bar -> the comparison is meaningful")

# ---------------------------------------------------------------- 2. the teacher's soft targets
log("\n--- generating the teacher's soft predictions for every TRAIN window ---")
t0 = time.perf_counter()
soft = np.zeros((len(Xtr), V), dtype=np.float32)
CH = 4096
for s in range(0, len(Xtr), CH):
    lg = teacher.forward_arrays(Xtr[s:s + CH])["logits"]
    soft[s:s + CH] = softmax_np(lg)
log(f"    {soft.shape} soft targets generated in {time.perf_counter()-t0:.1f}s")

# ---------------------------------------------------------------- 3. the two students
log(f"\n--- STUDENT-A (d={D_STUDENT}): real corpus + the teacher's soft predictions ---")
stud_a = build(D_STUDENT, SEED)
t_a = train_plain(stud_a, EPOCHS, soft_targets=soft, lam_soft=LAM_SOFT, tag="student-A")
ap, aa = evaluate(stud_a, Xva, yva)

log(f"\n--- STUDENT-B (d={D_STUDENT}): real corpus ONLY (the control, identical in every other way) ---")
stud_b = build(D_STUDENT, SEED)
t_b = train_plain(stud_b, EPOCHS, tag="student-B")
bp, ba = evaluate(stud_b, Xva, yva)

# ---------------------------------------------------------------- 4. verdict
log("\n" + "=" * 100)
log("RESULT")
log("=" * 100)
log(f"  {'model':<34}{'params':>11}{'ppl':>10}{'acc':>9}")
for tag, net, pp, aa_ in [("TEACHER  d=512 real", teacher, tp, ta),
                          ("STUDENT-A d=128 + teacher", stud_a, ap, aa),
                          ("STUDENT-B d=128 control", stud_b, bp, ba)]:
    log(f"  {tag:<34}{sum(p.data.size for p in net.parameters()):>11,}{pp:>10.3f}{aa_*100:>8.2f}%")

log(f"\n  STUDENT-A vs STUDENT-B (same size, same data, same seed, same steps):")
log(f"    accuracy  {aa*100:.2f}%  vs  {ba*100:.2f}%   ->  {aa-ba:+.4f}")
log(f"    perplexity {ap:.3f}  vs  {bp:.3f}   ->  {ap-bp:+.4f} (lower is better)")
better = (aa > ba) or (aa == ba and ap < bp)
log(f"\n  VERDICT: {'THE DISTILLED STUDENT IS BETTER' if better else 'THE DISTILLED STUDENT IS NOT BETTER'}")

res = {"verdict": "A better" if better else "A not better",
       "teacher": {"ppl": tp, "acc": ta, "params": sum(p.data.size for p in teacher.parameters())},
       "student_a": {"ppl": ap, "acc": aa, "params": sum(p.data.size for p in stud_a.parameters())},
       "student_b": {"ppl": bp, "acc": ba, "params": sum(p.data.size for p in stud_b.parameters())},
       "delta_acc": aa - ba, "delta_ppl": ap - bp, "lam_soft": LAM_SOFT,
       "teacher_bar": TEACHER_BAR, "epochs": EPOCHS}
with open(os.path.join(OUT, "verdict.json"), "w") as f:
    json.dump(res, f, indent=1)
lens.save(stud_a, os.path.join(OUT, "student_a.npz"))
lens.save(stud_b, os.path.join(OUT, "student_b.npz"))
lens.save(teacher, os.path.join(OUT, "teacher.npz"))
log(f"\nsaved {OUT}/verdict.json and the three checkpoints")
log("=" * 100)
