"""lens/scripts/distillation_test.py - a REAL distillation test: a teacher that DIFFERS from the labels.

WHY. UNIFIED-TRAINING.md records the failure exactly: the first teacher-mode run produced numbers IDENTICAL
to data mode "because the test teacher returned exactly a+b, the same as the labels -- an invalid distillation
test. A real test needs a teacher that differs from the labels."

WHAT A VALID TEST LOOKS LIKE. Same corpus, same loop, same steps. The labels say one thing and the teacher says
another; then check WHICH ONE the model ended up matching. If the mode genuinely selects the target source,
the teacher-mode model matches the TEACHER and not the labels, and the data-mode model does the opposite. If
both modes match the labels, the loop is ignoring the teacher and the "mode" is decoration.

TWO TEACHERS, so the test can also fail in the honest direction:
  linear teacher   t = a + b + 3          the model CAN represent it (its readout is linear) -> should be learned
  non-linear teacher t = a * b            the model CANNOT represent it (no multiplicative term) -> must NOT be
                                          learned; if this one also "succeeds", the test is measuring nothing

PREDICTION, WRITTEN BEFORE RUNNING (P12/P13)
  P12 teacher mode with t = a+b+3 ends matching the teacher on held-out pairs > 90% and the labels ~ 0%,
      while data mode ends matching the labels > 90% and the teacher ~ 0%. Both must be measured on pairs the
      model never saw, and on the model's own probe (never trained on).
  P13 the non-linear teacher t = a*b is NOT learned: teacher-match on held-out stays low (I expect under 20%),
      because the model has no way to represent a product. A test where every teacher passes is worthless.

FIRST RUN OF THIS FILE, SAME DAY: P12 FAILED. Teacher mode matched the teacher 6.7% and the labels 14.7% on
held-out - it learned neither, while data mode reached 90.7%. Cause found in lens/unified.py: in teacher mode
the model was trained to match the teacher while its own probe (and therefore the gap gate) was still scored
against the LABELS. Fixed: the probe is now scored against the same target the model is trained on.
  P14 (written before re-running, after that fix): teacher mode now matches the teacher >90% on held-out and
      the labels ~0%, so P12 becomes PASS; P13 stays PASS (a*b is still not representable).
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lens.unified import train_unified  # noqa: E402

STEPS = 3000
LR = 0.05
SEEDS = (0, 1, 2)


def corpus():
    pairs = [(a, b) for a in range(10) for b in range(10)]
    X = np.asarray([[a / 20.0, b / 20.0, 1.0] for a, b in pairs], dtype=np.float64)
    labels = np.asarray([a + b for a, b in pairs], dtype=np.float64)
    return X, labels, np.asarray(pairs, dtype=np.float64)


def match_rate(pred, target):
    return float((np.round(pred) == np.round(target)).mean())


def main():
    X, labels, pairs = corpus()
    print("=" * 100)
    print("REAL DISTILLATION TEST - a teacher that DIFFERS from the labels")
    print("=" * 100)
    print("P12: teacher=a+b+3 -> held-out matches the teacher >90% and the labels ~0%; data mode the reverse")
    print("P13: teacher=a*b (not representable by this model) must NOT be learned; teacher-match stays low")
    print("all numbers below: exact-match rate (rounded), mean over 3 seeds, on the model's own held-out probe")
    print()

    linear_teacher = lambda Xb: (Xb[:, 0] * 20.0 + Xb[:, 1] * 20.0 + 3.0)
    nonlinear_teacher = lambda Xb: ((Xb[:, 0] * 20.0) * (Xb[:, 1] * 20.0))

    rows = []
    print(f"{'mode':<22}{'teacher':<12}{'held-out vs teacher':>21}{'held-out vs labels':>20}"
          f"{'train vs teacher':>18}")
    print("-" * 100)
    for mode, teacher, tname in (("data", None, "-"),
                                 ("teacher", linear_teacher, "a+b+3"),
                                 ("teacher", nonlinear_teacher, "a*b")):
        h_t, h_l, tr_t = [], [], []
        for seed in SEEDS:
            (w, b), log = train_unified(X, labels, mode=mode, teacher=teacher, steps=STEPS, lr=LR,
                                        seed=seed, verbose=False)
            pred_all = X @ w + b
            t_all = labels if teacher is None else teacher(X)
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(X))
            n_held = max(8, int(len(X) * 0.25))
            hi = order[:n_held]
            p_h = pred_all[hi]
            h_t.append(match_rate(p_h, t_all[hi]))
            h_l.append(match_rate(p_h, labels[hi]))
            tr_t.append(match_rate(pred_all, t_all))
        rows.append({"mode": mode, "teacher": tname,
                     "heldout_vs_teacher_mean": float(np.mean(h_t)), "heldout_vs_teacher_std": float(np.std(h_t)),
                     "heldout_vs_labels_mean": float(np.mean(h_l)), "heldout_vs_labels_std": float(np.std(h_l)),
                     "train_vs_teacher_mean": float(np.mean(tr_t))})
        print(f"{mode:<22}{tname:<12}{np.mean(h_t)*100:>15.1f}%+-{np.std(h_t)*100:4.1f}"
              f"{np.mean(h_l)*100:>15.1f}%+-{np.std(h_l)*100:4.1f}{np.mean(tr_t)*100:>17.1f}%")
    print("-" * 100)

    # ---- verdicts, against P12/P13
    # NOTE ON THE VERDICT ITSELF: the data row has no teacher, so its "vs teacher" column is the labels by
    # construction and cannot be tested against a teacher. The first version of this check required
    # data-mode heldout_vs_teacher < 0.10, which can never hold, and so reported FAIL on a passing result.
    # Found by reading the row instead of the verdict word. The data row is now judged only on the labels.
    lin = [r for r in rows if r["teacher"] == "a+b+3"][0]
    dat = [r for r in rows if r["mode"] == "data"][0]
    non = [r for r in rows if r["teacher"] == "a*b"][0]
    v12 = (lin["heldout_vs_teacher_mean"] > 0.90 and lin["heldout_vs_labels_mean"] < 0.10
           and dat["heldout_vs_labels_mean"] > 0.90)
    v13 = non["heldout_vs_teacher_mean"] < 0.20
    print(f"\nP12  teacher mode matches the teacher {lin['heldout_vs_teacher_mean']*100:.1f}% and the labels "
          f"{lin['heldout_vs_labels_mean']*100:.1f}%; data mode the labels {dat['heldout_vs_labels_mean']*100:.1f}% "
          f"(no teacher to match)          -> {'PASS' if v12 else 'FAIL'}")
    print(f"P13  non-representable teacher a*b matched {non['heldout_vs_teacher_mean']*100:.1f}% "
          f"(must stay under 20%)                                              -> {'PASS' if v13 else 'FAIL'}")
    print("\nA FAIL on P12 means the mode is decoration; a FAIL on P13 means the test proves nothing.")


if __name__ == "__main__":
    main()
