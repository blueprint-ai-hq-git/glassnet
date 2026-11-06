"""g_loop.py — Target G: self-improvement with a checkable answer.

Pipeline (pre-registered in TARGET-G.md before this ran):

  1. teacher        : trained from scratch on a fixed 70 of the 100 problems
  2. generate       : the teacher answers ALL 100 problems, 20 samples each, stochastic
  3. verify         : integer arithmetic keeps only the CORRECT answers
  4. students       : same architecture, same budget, three arms
                        verified   -- the correct samples only
                        unverified -- the same NUMBER of raw samples (control)
                        real       -- real problems, same count (control)
  5. compare        : exact-answer accuracy on the 30 HELD-OUT problems

If `verified` beats the teacher but `unverified` does too, the win is from extra data and G-WIN
is VOID.
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)

import numpy as np  # noqa: E402

from glassnet import arith                     # noqa: E402
from glassnet.arith_model import ArithNet, train  # noqa: E402
from glassnet.tensor import set_backend        # noqa: E402

set_backend("numpy")

R = os.path.join(PROJ, "runs", "t3")
SEEDS = [int(x) for x in os.environ.get("G_SEEDS", "0,1,2,3,4").split(",")]
EPOCHS = int(os.environ.get("G_EPOCHS", "60"))
LR = float(os.environ.get("G_LR", "1.0"))
SAMPLES_PER_PROBLEM = int(os.environ.get("G_SAMPLES", "20"))
TEMPERATURE = float(os.environ.get("G_TEMP", "1.5"))
TEACHER_N = int(os.environ.get("G_TEACHER_N", "70"))
D, K = 64, 8

ALL = arith.problems()


def make_pairs(problems, rng=None, repeat_to=None):
    """Build the two-step (window, target) pairs for a list of problems."""
    Xs, ys = [], []
    for a, b in problems:
        s = arith.encode(a, b)
        Xs.append(np.array([arith.TOKEN[arith.PAD], arith.TOKEN[arith.BOS], s[0], s[1], s[2], s[3]]))
        ys.append(s[4])
        Xs.append(np.array([arith.TOKEN[arith.BOS], s[0], s[1], s[2], s[3], s[4]]))
        ys.append(s[5])
    X, y = np.array(Xs), np.array(ys)
    if repeat_to is not None and repeat_to > len(X):
        idx = np.resize(np.arange(len(X)), repeat_to)
        X, y = X[idx], y[idx]
    return X, y


def make_pairs_from_seqs(seqs):
    """Same two-step pairs, but from GENERATED sequences (so wrong answers stay wrong)."""
    Xs, ys = [], []
    for s in seqs:
        Xs.append(np.array([arith.TOKEN[arith.PAD], arith.TOKEN[arith.BOS], s[0], s[1], s[2], s[3]]))
        ys.append(s[4])
        Xs.append(np.array([arith.TOKEN[arith.BOS], s[0], s[1], s[2], s[3], s[4]]))
        ys.append(s[5])
    return np.array(Xs), np.array(ys)


def concept_labels(X):
    return np.stack([[fn(int(x[1]), int(x[3])) for _, fn in arith.CONCEPTS] for x in X], axis=1)


print("=" * 100)
print("TARGET G — SELF-IMPROVEMENT WITH A CHECKABLE ANSWER")
print("=" * 100)
print(f"seeds {SEEDS} | teacher trains on {TEACHER_N}/100 | {SAMPLES_PER_PROBLEM} samples per "
      f"problem | temperature {TEMPERATURE} | {EPOCHS} epochs | lr {LR}")
print(f"vocab {arith.V} | window 6 | concepts {[n for n, _ in arith.CONCEPTS]}")

results = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ALL))
    teacher_probs = [ALL[i] for i in sorted(perm[:TEACHER_N])]
    holdout = [ALL[i] for i in sorted(perm[TEACHER_N:])]

    # ---------------------------------------------------------- 1. teacher
    Xt, yt = make_pairs(teacher_probs)
    teacher = ArithNet(k=K, d=D, seed=seed, readable=False)
    train(teacher, Xt, yt, epochs=EPOCHS, lr=LR, seed=seed)
    t_train = teacher.accuracy(teacher_probs)
    t_hold = teacher.accuracy(holdout)

    # ---------------------------------------------------------- 2. generate
    gen = []
    for a, b in ALL:
        for _ in range(SAMPLES_PER_PROBLEM):
            t, u = teacher.sample_problem(a, b, rng, temperature=TEMPERATURE)
            gen.append([arith.TOKEN[str(a)], arith.TOKEN[arith.PLUS], arith.TOKEN[str(b)],
                        arith.TOKEN[arith.EQ], t, u])
    gen = np.array(gen)
    ok = arith.is_correct(gen)
    verified = gen[ok]
    n_verify = len(verified)
    unverified = gen[:max(n_verify, 1)]
    # how many DISTINCT problems have at least one verified answer?
    covered = len({(int(s[0]), int(s[2])) for s in verified})

    # ---------------------------------------------------------- 3. students
    arms = {}
    Xv, yv = make_pairs_from_seqs(verified)
    sv = ArithNet(k=K, d=D, seed=seed + 1000, readable=False)
    train(sv, Xv, yv, epochs=EPOCHS, lr=LR, seed=seed)
    arms["verified"] = sv.accuracy(holdout)

    Xu, yu = make_pairs_from_seqs(unverified)
    su = ArithNet(k=K, d=D, seed=seed + 1000, readable=False)
    train(su, Xu, yu, epochs=EPOCHS, lr=LR, seed=seed)
    arms["unverified"] = su.accuracy(holdout)

    n_pairs = max(len(Xv), 2)
    Xr, yr = make_pairs(ALL, repeat_to=n_pairs)
    sr = ArithNet(k=K, d=D, seed=seed + 1000, readable=False)
    train(sr, Xr, yr, epochs=EPOCHS, lr=LR, seed=seed)
    arms["real"] = sr.accuracy(holdout)

    print(f"\nseed {seed}:")
    print(f"  teacher        train-set acc {t_train*100:6.2f}%   HELD-OUT {t_hold*100:6.2f}%")
    print(f"  generated      {len(gen)} samples -> {n_verify} verified "
          f"({n_verify/len(gen)*100:.1f}%), covering {covered}/100 problems")
    print(f"  student verified   held-out {arms['verified']*100:6.2f}%")
    print(f"  student unverified held-out {arms['unverified']*100:6.2f}%   (control)")
    print(f"  student real       held-out {arms['real']*100:6.2f}%   (control)")

    results.append({"seed": seed, "teacher_train": t_train, "teacher_holdout": t_hold,
                    "n_generated": len(gen), "n_verified": n_verify,
                    "problems_covered": covered, "arms": arms})

# ------------------------------------------------------------------ verdict
print("\n" + "=" * 100)
print("VERDICT — pre-registered bars: G-WIN = verified beats teacher by > 2x pooled std (5 seeds); "
      "G-CONTROL = unverified must NOT")
print("=" * 100)


def stats(vals):
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


th_m, th_s = stats([r["teacher_holdout"] for r in results])
v_m, v_s = stats([r["arms"]["verified"] for r in results])
u_m, u_s = stats([r["arms"]["unverified"] for r in results])
r_m, r_s = stats([r["arms"]["real"] for r in results])
pooled_vt = float(np.sqrt((th_s ** 2 + v_s ** 2) / 2))
pooled_ut = float(np.sqrt((th_s ** 2 + u_s ** 2) / 2))

print(f"  teacher          {th_m*100:6.2f}% +/- {th_s*100:.2f}")
print(f"  student verified {v_m*100:6.2f}% +/- {v_s*100:.2f}   delta {(v_m-th_m)*100:+.2f} pts "
      f"(bar {2*pooled_vt*100:.2f})")
print(f"  student unverif. {u_m*100:6.2f}% +/- {u_s*100:.2f}   delta {(u_m-th_m)*100:+.2f} pts "
      f"(bar {2*pooled_ut*100:.2f})")
print(f"  student real     {r_m*100:6.2f}% +/- {r_s*100:.2f}   delta {(r_m-th_m)*100:+.2f} pts")

g_win = (v_m - th_m) > 2 * pooled_vt
g_control = (u_m - th_m) > 2 * pooled_ut
print(f"\n  G-WIN      : {'PASS' if g_win else 'FAIL'}  (verified beats teacher by more than "
      f"2x pooled std)")
print(f"  G-CONTROL  : {'PASS' if not g_control else 'FAIL'}  (unverified must NOT beat it; "
      f"if it does, G-WIN is VOID)")
if g_win and g_control:
    print("  => G-WIN is VOID: the gain is from extra data, not from verification.")
elif g_win:
    print("  => WIN, and the control holds: verification is doing the work.")
else:
    print("  => NO WIN. The teacher was not beaten. This is the result.")

json.dump({"config": {"seeds": SEEDS, "epochs": EPOCHS, "lr": LR, "samples": SAMPLES_PER_PROBLEM,
                      "temperature": TEMPERATURE, "teacher_n": TEACHER_N, "d": D, "k": K},
           "runs": results,
           "summary": {"teacher_holdout": [th_m, th_s], "verified": [v_m, v_s],
                       "unverified": [u_m, u_s], "real": [r_m, r_s],
                       "g_win": bool(g_win), "g_control_beaten": bool(g_control)}},
          open(os.path.join(R, "g_loop.json"), "w"), indent=1)
print(f"\nsaved runs/t3/g_loop.json")
