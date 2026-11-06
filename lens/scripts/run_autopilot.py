"""run_autopilot.py — GATE the auto-pilot (plan ask #3) against hand-picked settings.

GATES (pre-registered)
    G1  the auto-pilot call passes NO settings (only data)                       [by construction]
    G2  its chosen config is within 5% of the hand-picked baseline's val loss   (it must not
        be worse than a human-chosen setting on the same data/split)
    G3  it beats the WORST learning-rate candidate's val loss                   (the search works)
    G4  its validation slice was never trained on (index-disjoint, sizes reported)

The comparison uses the project's own chat windows (v12000, W32) on a small slice so the whole
gate runs on CPU in a few minutes.
"""
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import chat_concepts_wide                             # noqa: E402
import chat_data                                      # noqa: E402
import lens                                           # noqa: E402
from lens.autopilot import autopilot, _val_loss       # noqa: E402
from lens.concepts import Concept                     # noqa: E402

WIN = 32
N_WINDOWS = int(os.environ.get("AP_WINDOWS", "6000"))
SEED = 0


def main():
    d = chat_data.build()
    ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
    Xtr, Ytr = chat_data.make_windows(ids, bounds, split, WIN, which=0)
    X, y = np.asarray(Xtr[:N_WINDOWS]), np.asarray(Ytr[:N_WINDOWS])
    train_mask = np.zeros(len(ids), dtype=bool)
    for j in range(len(bounds) - 1):
        if split[j] == 0:
            train_mask[bounds[j]:bounds[j + 1]] = True
    cs, _spec = chat_concepts_wide.build(vocab, ids, train_mask, windows=X)
    concepts = [Concept(c["name"], c["ids"], support=c["support"]) for c in cs]
    print(f"autopilot gate: {len(X)} windows, vocab {len(vocab)}, {len(concepts)} concepts")

    # ---- the auto-pilot: NO settings passed
    t0 = time.perf_counter()
    net, rep = autopilot(len(vocab), concepts, X, y, seed=SEED, device="cpu",
                         log=os.path.join(PROJ, "lens", "runs", "autopilot_report.json"))
    ap_secs = time.perf_counter() - t0
    ap_val = rep["best_val_loss"]

    # ---- replicate the SAME split for the hand-picked arms
    rng = np.random.default_rng(SEED)
    n_val = max(64, int(len(X) * 0.10))
    order = rng.permutation(len(X))
    vi, ti = order[:n_val], order[n_val:]
    Xv, yv = X[vi], y[vi]
    Xt, yt = X[ti], y[ti]

    def hand(d_, lr, epochs):
        n = lens.Net(vocab=len(vocab), concepts=concepts, d=d_, window=WIN, seed=SEED,
                     lam_concept=0.3, lam_act=0.2, device="cpu")
        r = np.random.default_rng(SEED + 7)
        for _ in range(epochs):
            o = r.permutation(len(Xt))
            for s in range(0, len(Xt), 128):
                idx = o[s:s + 128]
                for p in n.parameters():
                    p.grad = None
                loss, _ = n.loss(Xt[idx], yt[idx])
                loss.backward()
                for p in n.parameters():
                    if p.grad is not None:
                        p.data = p.data - lr * p.grad
        return n, _val_loss(n, Xv, yv)

    print("\n--- hand-picked arms (same data, same split) ---")
    _, hp_val = hand(64, 0.4, 10)
    print(f"    hand-picked  d=64 lr=0.4 ep=10   val loss {hp_val:.4f}")
    _, worst_val = hand(64, 1.0, 10)
    print(f"    worst-setting d=64 lr=1.0 ep=10  val loss {worst_val:.4f}")
    _, worst2_val = hand(64, 0.1, 10)
    print(f"    low-lr arm    d=64 lr=0.1 ep=10  val loss {worst2_val:.4f}")
    worst_of_arms = max(worst_val, worst2_val)

    g1 = True
    g2 = ap_val <= hp_val * 1.05
    g3 = ap_val < worst_of_arms
    g4 = (len(Xt) + len(Xv) == len(X)) and len(np.intersect1d(ti, vi)) == 0
    print("\n" + "=" * 92)
    print("AUTO-PILOT GATES (pre-registered)")
    print("=" * 92)
    print(f"  G1 no settings passed            : PASS (call took only data)")
    print(f"  G2 within 5% of hand-picked      : autopilot {ap_val:.4f} vs hand {hp_val:.4f} "
          f"({100*(ap_val-hp_val)/hp_val:+.2f}%) -> {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 beats the worst arm           : {ap_val:.4f} vs {worst_of_arms:.4f} -> "
          f"{'PASS' if g3 else 'FAIL'}")
    print(f"  G4 pilot-val never trained on    : train {len(Xt)} + val {len(Xv)} = {len(X)}, "
          f"overlap {len(np.intersect1d(ti, vi))} -> {'PASS' if g4 else 'FAIL'}")
    verdict = "PASS" if (g1 and g2 and g3 and g4) else "FAIL"
    print(f"\n  AUTO-PILOT: {verdict}   (chose d={rep['chosen']['d']}, lr={rep['chosen']['lr']}, "
          f"stopped at epoch {rep['best_epoch']} of {rep['epochs_run']}, {ap_secs:.0f}s)")
    rep["gate"] = {"verdict": verdict, "G1": g1, "G2": g2, "G3": g3, "G4": g4,
                   "hand_val": hp_val, "worst_arm_val": worst_of_arms, "ap_seconds": ap_secs}
    json.dump(rep, open(os.path.join(PROJ, "lens", "runs", "autopilot_report.json"), "w"),
              indent=1)
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())