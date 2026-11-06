"""audit_strict_vision.py — strict label-honesty for the SEE+TALK (K22) ship.

Mirrors audit_strict_text.py exactly, for images: every concept neuron h[:,k] is correlated
with EVERY concept formula's incidence (not just its own) on the SAME val windows the ship's
gates used (datum reproduced deterministically from the run's own saved args), and a neuron is
honest only if |own| beats |best-other| by > 0.10. AUROC per concept reported too (g2-style).

Exit 0 = 22/22 honest (100%), 2 = gaps named.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import train_seetalk as TS                     # noqa: E402
from lens import backend as B                  # noqa: E402


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def auroc(scores, y):
    """Area under the ROC via the rank-sum (Mann-Whitney) identity — robust to ties.
    AUC = (sum of average ranks of positives - n_p(n_p+1)/2) / (n_p*n_n)."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n_p = int(y.sum())
    n_n = len(y) - n_p
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    srt = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0          # average rank for the tie block
        i = j + 1
    r_pos = float(ranks[np.where(y[order] == 1)].sum())
    return float((r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def main():
    ship = os.environ.get("VISION_SHIP", "seetalk_k22_big_fixedgrad.npz")
    ckpt = os.environ.get("VISION_CKPT") or os.path.join(PROJ, "lens", "runs", "seetalk", ship)
    lj = ckpt[:-4] + "_log.json"
    if not os.path.exists(ckpt):
        print(f"no ship at {ckpt}"); return 3
    args = json.load(open(lj))["args"] if os.path.exists(lj) else {}
    # The datum size must be COMPARABLE across ships. A ship whose log json was never written
    # (a crash before the log dump) used to silently fall back to n=4000 while every other ship
    # was audited at n=60000 -- the same measurement at a different sample size is not the same
    # number. VISION_AUDIT_N pins it explicitly, and the value used is printed.
    n = int(os.environ.get("VISION_AUDIT_N", args.get("n", 4000)))
    # THE DATUM IS THE MEASURING INSTRUMENT: its seed is PINNED to 0 regardless of the model's
    # init seed. Reading the ship's own seed here (2026-10-13) made every seed-varied ship get
    # audited on DIFFERENT windows, so the strict numbers were not comparable across runs —
    # the wrong-subject class again. Only n and img_start may vary with the ship.
    seed = 0
    istart = int(args.get("img_start", 0))
    print(f"  ship {ship}  datum args: n={n}, seed={seed} (PINNED), img_start={istart}"
          f"  [ship init seed {args.get('seed')}]")

    data = TS.build_caption_data(n=n, seed=seed, img_start=istart)
    net, meta = TS.SeeTalkNet.load(ckpt)
    K = len(net.concepts)
    Xv = data["Xva"]
    print(f"  concepts {K} (meta V={meta['V']} d={meta['D']} window={meta['W']})  "
          f"val windows {len(Xv)}")

    # h = image -> concept activations (h depends only on the image; cap_ids irrelevant)
    Pn = net.front.patches(Xv)
    zp = Pn.reshape(-1, Pn.shape[-1]) @ B.to_cpu(net.front.proj.W.data) + B.to_cpu(net.front.proj.b.data)
    zflat = zp.reshape(len(Xv), -1)
    if getattr(net, "concept_direct", False):
        # the bypass: the concept layer also reads the raw patches -- mirror it here exactly.
        zflat = np.concatenate([zflat, Pn.reshape(len(Xv), -1)], axis=1)
    if getattr(net, "concept_mirror", False):
        # the |I - mirror(I)| block: raw material for a pairwise formula. Mirrored here exactly.
        zflat = np.concatenate([zflat, net._mirror_map(Xv)], axis=1)
    cl = zflat @ B.to_cpu(net.concept.W.data) + B.to_cpu(net.concept.b.data)
    h = 1.0 / (1.0 + np.exp(-np.clip(cl, -60, 60)))          # (N, K)

    # every formula on the same windows
    names = [c.name for c in net.concepts]
    L = np.stack([c.labels(Xv) for c in net.concepts], axis=1).astype(np.float64)

    M = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            M[i, j] = corr(h[:, i], L[:, j])

    rows = []
    for i in range(K):
        own = M[i, i]
        best_other_abs, j_best = max((abs(M[i, j]), j) for j in range(K) if j != i)
        best_other = M[i, j_best]
        rows.append({"concept": names[i], "own_corr": round(float(own), 3),
                     "best_other": round(float(best_other), 3),
                     "best_other_name": names[j_best],
                     "margin": round(float(own) - best_other_abs, 3),
                     "honest": bool(own - best_other_abs > 0.10),
                     "mislabel": bool(best_other_abs - own > 0.10),
                     "auroc": round(auroc(h[:, i], L[:, i]), 3)})

    print("\n  STRICT LABEL-HONESTY (val windows, |own| vs |best-other| margin > 0.10)")
    print(f"    {'concept':<26}{'own':>7}{'best-other':>12}{'other':<22}{'AUROC':>8}{'verdict':>12}")
    for r in sorted(rows, key=lambda r: r["auroc"], reverse=True):
        v = "MISLABELLED" if r["mislabel"] else ("honest" if r["honest"] else "weak")
        print(f"    {r['concept'][:25]:<26}{r['own_corr']:>7.3f}{r['best_other']:>12.3f}"
              f"  {r['best_other_name'][:20]:<22}{r['auroc']:>8.3f}{v:>12}")
    hh = sum(r["honest"] for r in rows)
    mm = sum(r["mislabel"] for r in rows)
    print(f"\n  STRICT: honest {hh}/{K} (100% bar) — {'PASS' if hh == K else 'FAIL'}; "
          f"mislabelled {mm}; AUROC<0.8: {sum(1 for r in rows if r['auroc'] < 0.8)}")
    json.dump({"ship": ship, "rows": rows, "honest": hh, "n": K},
              open(os.path.join(PROJ, "lens", "runs", "seetalk", "strict_vision_audit.json"), "w"),
              indent=1)
    return 0 if hh == K else 2


if __name__ == "__main__":
    sys.exit(main())