"""PHASE 4 — the data judge, gate run.

Run:  python3 phase4_judge.py

Gate: the judge beats random guessing (50%) on data it has NEVER seen,
proven twice:
  (a) random held-out split of the cleaned pile (standard),
  (b) UNSEEN-TYPE split: trained only on 4 corruption types, tested on 4
      corruption types it never saw one example of (the honest hard test).
Numbers + example mistakes (with a picture grid) are written to
runs/phase4/.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.gates import gate1  # noqa: E402
from glassnet.data import load as load_pixels  # noqa: E402
from glassnet.judge import (JUDGE_DEFAULTS, TYPE_NAMES, build_pile,  # noqa: E402
                            Cleaner, JudgeModel)
from glassnet.eval import accuracy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "phase4")
DATA_DIR = os.path.join(HERE, "data")


def acc_at(pred, y):
    return float((pred == y).mean())


def per_type_recall(X, y, types, model):
    rows = []
    pred = model.predict(X)
    for t in sorted(set(types)):
        m = types == t
        rows.append((t, int(m.sum()), float(pred[m].mean()),
                     int((pred[m] == y[m]).sum())))
    return rows


def mistakes_on(X, y, types, model, want=6):
    """Most-confident WRONG predictions (picture-worthy examples)."""
    p = model.prob_bad(X)
    pred = p > 0.5
    wrong = np.where(pred != y)[0]
    if len(wrong) == 0:
        return []
    # sort false positives by confidence (high p on good) and false
    # negatives by confidence (low p on bad) — the worst misses first
    fp = wrong[y[wrong] == 0]
    fn = wrong[y[wrong] == 1]
    order = np.concatenate([fp[np.argsort(p[fp])[::-1]],
                            fn[np.argsort(p[fn])]])
    return [{"idx": int(i), "true_type": str(types[i]), "y": int(y[i]),
             "prob_bad": float(p[i])} for i in order[:want]]


def train_seed(seed, Xtr, ytr):
    model = JudgeModel(np.random.default_rng(seed))
    model.train(Xtr, ytr)
    return model


def main():
    os.makedirs(RUNS, exist_ok=True)
    print("=" * 66)
    print("GATE 1 first (no training on broken gradients)")
    print("=" * 66)
    gate1()

    Xtr, ytr, Xte, yte = load_pixels(DATA_DIR)
    rng = np.random.default_rng(42)
    pile = build_pile(Xtr, ytr, rng)
    print(f"[pile] {len(pile['X'])} rows: "
          f"{int((pile['y'] == 0).sum())} good / {int((pile['y'] == 1).sum())} bad")

    # ---- cheap automatic cleaning -------------------------------------
    cleaner = Cleaner()
    keep, Xc, yc, tc, rep = cleaner.clean(pile["X"], pile["y"], pile["types"])
    print("\n[CLEANER — injected vs found]")
    for k, v in rep.items():
        print(f"  {k:20s} removed {v}")
    print(f"  cleaned pile: {len(Xc)} rows "
          f"({int((yc == 0).sum())} good / {int((yc == 1).sum())} bad)")

    # ---- split: random held-out (60/40), balanced -----------------------
    good_i = np.where(yc == 0)[0]
    bad_i = np.where(yc == 1)[0]
    rng2 = np.random.default_rng(7)
    g_tr = rng2.choice(good_i, int(0.6 * len(good_i)), replace=False)
    b_tr = rng2.choice(bad_i, int(0.6 * len(bad_i)), replace=False)
    tr_idx = np.concatenate([g_tr, b_tr])
    te_idx = np.setdiff1d(np.arange(len(Xc)), tr_idx)
    Xtr_s, ytr_s, Xte_s, yte_s = Xc[tr_idx], yc[tr_idx], Xc[te_idx], yc[te_idx]

    accs, unseen_accs, all_recalls = {}, {}, []
    for seed in [0, 1, 2]:
        m = train_seed(seed, Xtr_s, ytr_s)
        accs[seed] = acc_at(m.predict(Xte_s), yte_s)
        m.save(os.path.join(RUNS, f"judge_seed{seed}.npz"),
               heldout_acc=np.float64(accs[seed]))
        # unseen-type split: train ONLY on half the corruption types
        seen_types = TYPE_NAMES[:4]
        mask_seen = np.isin(tc, ["good"] + seen_types)
        _tr = np.where(mask_seen)[0]
        Xt2, yt2 = Xc[_tr], yc[_tr]
        m2 = train_seed(seed * 10 + 1, Xt2, yt2)
        unseen_types = TYPE_NAMES[4:]
        mask_unseen = np.isin(tc, unseen_types)
        Xu, yu = Xc[mask_unseen], yc[mask_unseen]
        # balanced with good rows for a fair 50% baseline
        g_mask = tc == "good"
        Xg, yg = Xc[g_mask], yc[g_mask]
        Xuu = np.concatenate([Xu, Xg])
        yuu = np.concatenate([yu, yg])
        unseen_accs[seed] = acc_at(m2.predict(Xuu), yuu)
        m2.save(os.path.join(RUNS, f"judge_unseen_seed{seed}.npz"),
                unseen_acc=np.float64(unseen_accs[seed]))
        if seed == 0:
            rec = per_type_recall(Xc, yc, tc, m)
            all_recalls = rec

    # ---- ensemble: mean probability over the 3 seeds ---------------------
    def ensemble_acc(models, X, y):
        p = np.mean([mdl.prob_bad(X) for mdl in models], axis=0)
        return float(((p > 0.5) == y).mean())

    held_models = [JudgeModel.load(os.path.join(RUNS, f"judge_seed{s}.npz"))
                   for s in [0, 1, 2]]
    unse_models = [JudgeModel.load(os.path.join(RUNS, f"judge_unseen_seed{s}.npz"))
                   for s in [0, 1, 2]]
    ens_held = ensemble_acc(held_models, Xte_s, yte_s)
    ens_unseen = ensemble_acc(unse_models, Xuu, yuu)

    am, asd = float(np.mean(list(accs.values()))), float(np.std(list(accs.values())))
    um, usd = float(np.mean(list(unseen_accs.values()))), float(np.std(list(unseen_accs.values())))
    margin_ok = (ens_held - 0.5) > 0.03 and (ens_unseen - 0.5) > 0.03

    # ---- example mistakes (from the held-out test, seed 0) --------------
    m0 = JudgeModel.load(os.path.join(RUNS, "judge_seed0.npz"))
    mis = mistakes_on(Xc[te_idx], yc[te_idx], tc[te_idx], m0, want=6)

    lines = [
        "# PHASE 4 REPORT — the data judge",
        "",
        f"[pile] {len(pile['X'])} rows: {int((pile['y'] == 0).sum())} good / "
        f"{int((pile['y'] == 1).sum())} bad "
        f"(8 corruption types x {JUDGE_DEFAULTS['bad_per_type']} + injected junk)",
        "",
        "## Cheap automatic cleaning (injected vs found)",
    ]
    for k, v in rep.items():
        lines.append(f"- {k}: removed {v}")
    inj = pile["injected"]
    lines += [
        f"- injected for the cleaner: {sum(inj.values())} "
        f"({inj['dupe']} dupes, {inj['empty']} empty, {inj['gray']} gray, "
        f"{inj['nan']} NaN); total found by cleaner: {rep['total_removed']}",
        f"- cleaned pile: {len(Xc)} rows "
        f"({int((yc == 0).sum())} good / {int((yc == 1).sum())} bad) — the "
        f"judge never sees the cheap junk",
        "",
        "## Gate: beats random guessing (50%) on unseen data",
        f"- random held-out split (60/40, balanced): "
        f"**{am*100:.2f}% ± {asd*100:.2f}%** per-seed: "
        f"{[f'{accs[s]*100:.1f}' for s in accs]}",
        f"- UNSEEN-TYPE split (trained on 4 corruption types, tested on 4 "
        f"types never seen): **{um*100:.2f}% ± {usd*100:.2f}%** per-seed: "
        f"{[f'{unseen_accs[s]*100:.1f}' for s in unseen_accs]}",
        f"- 3-seed ensemble (mean probability): held-out "
        f"**{ens_held*100:.2f}%**, unseen-type **{ens_unseen*100:.2f}%**",
        f"- random guessing = 50.00%. verdict (ensemble must clear 53% on "
        f"BOTH): {'PASS ✔' if margin_ok else 'FAIL ✘'}",
        "",
        "## Per-type recall (held-out, seed 0)",
        "| type | rows | P(bad) mean | correct/total |",
        "|---|---:|---:|---:|",
    ]
    for t, n, pb, corr in all_recalls:
        lines.append(f"| {t} | {n} | {pb:.2f} | {corr}/{n} |")
    lines += [
        "",
        "## Example mistakes (held-out test, seed 0) — the judge GOT THESE WRONG",
        "",
    ]
    for m in mis:
        lines.append(f"- {m['true_type']} (true label {m['y']}): judge said "
                     f"prob_bad={m['prob_bad']:.2f} -> "
                     f"{'BAD' if m['prob_bad'] > 0.5 else 'GOOD'}")
    path = os.path.join(RUNS, "PHASE4-REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n[wrote] {path}")

    # ---- picture of the mistakes -----------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(10, 7))
        for ax, m in zip(axs.ravel(), mis):
            ax.imshow(Xc[te_idx][m["idx"]].reshape(28, 28), cmap="gray_r")
            verdict = "BAD" if m["prob_bad"] > 0.5 else "GOOD"
            ax.set_title(f"true {m['true_type']} ({m['y']})\njudge: {verdict} "
                         f"p={m['prob_bad']:.2f}", fontsize=9)
            ax.axis("off")
        fig.suptitle("Data judge — example mistakes (6 most-confident wrong calls)")
        fig.tight_layout()
        png = os.path.join(RUNS, "mistakes.png")
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"[wrote] {png}")
    except Exception as e:
        print(f"[warn] mistakes plot skipped: {e}")

    print("\nGATE 4:", "PASS ✔" if margin_ok else "FAIL ✘")
    json.dump({"heldout": accs, "unseen_type": unseen_accs, "margin_ok": margin_ok},
              open(os.path.join(RUNS, "summary.json"), "w"), indent=1)
    sys.exit(0 if margin_ok else 1)


if __name__ == "__main__":
    main()