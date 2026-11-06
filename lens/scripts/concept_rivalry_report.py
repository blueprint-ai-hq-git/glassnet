"""concept_rivalry_report.py — WHY the rivalry penalty failed: measure the thing it was supposed
to change, on the same datum the strict audit uses.

It reuses the audit's exact code path (same datum builder, same front/concept arithmetic, same
formula labels) and prints, per ship:
  * mean |corr| between the concept ACTIVATIONS (what the penalty attacks)
  * pairs of activations above 0.8
  * the audit's own view: mean own |corr|, mean best-rival |corr|, and the honest count

If the penalty did its job (activations decorrelated) and the audit still got worse, the lever
worked as designed and the goal did not follow -- that is the honest finding to report.
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


def measure(ship, n=60000, seed=0, istart=0):
    ckpt = os.path.join(PROJ, "lens", "runs", "seetalk", ship)
    if not os.path.exists(ckpt):
        return None
    data = TS.build_caption_data(n=n, seed=seed, img_start=istart)
    net, meta = TS.SeeTalkNet.load(ckpt)
    Xv = data["Xva"]
    Pn = net.front.patches(Xv)
    zp = (Pn.reshape(-1, Pn.shape[-1]) @ B.to_cpu(net.front.proj.W.data)
          + B.to_cpu(net.front.proj.b.data))
    _zd = zp.reshape(len(Xv), -1)
    if getattr(net, "concept_direct", False):
        _zd = np.concatenate([_zd, Pn.reshape(len(Xv), -1)], axis=1)
    if getattr(net, "concept_mirror", False):
        _zd = np.concatenate([_zd, net._mirror_map(Xv)], axis=1)
    cl = (_zd @ B.to_cpu(net.concept.W.data)
          + B.to_cpu(net.concept.b.data))
    h = 1.0 / (1.0 + np.exp(-np.clip(cl, -60, 60)))            # (N, K) concept activations
    L = np.stack([c.labels(Xv) for c in net.concepts], axis=1).astype(np.float64)
    K = h.shape[1]
    C = np.corrcoef(h.T)
    off = ~np.eye(K, dtype=bool)
    ac = np.abs(C[off])
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            M[i, j] = corr(h[:, i], L[:, j])
    own = np.abs(np.diag(M))
    rival = np.array([np.max(np.abs(np.delete(M[i], i))) for i in range(K)])
    honest = int((own - rival > 0.10).sum())
    return dict(K=K, mean_act_corr=float(ac.mean()), pairs_gt_08=int((ac > 0.8).sum() // 2),
                max_act_corr=float(ac.max()), mean_own=float(own.mean()),
                mean_rival=float(rival.mean()), mean_margin=float((own - rival).mean()),
                honest=honest, own=own, rival=rival, names=[c.name for c in net.concepts])


def main():
    ships = sys.argv[1:] or ["seetalk_k21d.npz", "seetalk_k21j.npz",
                             "seetalk_k21k_d0.3.npz", "seetalk_k21k_d1.0.npz"]
    print(f"{'ship':<24} {'mean|corr|':>10} {'>0.8':>5} {'max':>7} {'mean own':>9} "
          f"{'mean rival':>11} {'mean margin':>12} {'honest':>7}")
    for sh in ships:
        r = measure(sh)
        if r is None:
            print(f"  {sh:<22} (missing)")
            continue
        print(f"  {sh:<22} {r['mean_act_corr']:>10.4f} {r['pairs_gt_08']:>5} "
              f"{r['max_act_corr']:>7.3f} {r['mean_own']:>9.3f} {r['mean_rival']:>11.3f} "
              f"{r['mean_margin']:>12.4f} {r['honest']:>4}/{r['K']}")
        if os.environ.get("DETAIL"):
            for nm, o, rv in zip(r["names"], r["own"], r["rival"]):
                flag = "ok" if o - rv > 0.10 else "WEAK"
                print(f"      {nm:<28} own {o:.3f}  rival {rv:.3f}  margin {o - rv:+.3f}  {flag}")


if __name__ == "__main__":
    main()
