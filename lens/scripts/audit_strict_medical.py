"""audit_strict_medical.py — strict label-honesty for the MEDICAL (PathMNIST) line.

Mirrors audit_strict_vision.py for the colour-concept VL ship: every image concept neuron
h[:,k] is correlated with EVERY image concept formula's incidence on the same test datum the
ship's gates used (reproduced from the run's saved args), and a neuron is honest only if
|own| beats |best-other| by > 0.10. AUROC per concept too. Text concepts are reported by
their run-log audit (AUROC) since their formulas are question-token id sets.

Exit 0 = n/n honest, 2 = gaps named (weak/mislabelled).
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

from lens import backend as B            # noqa: E402
from lens.vision import load_vision      # noqa: E402


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def auroc(scores, y):
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n_p = int(y.sum())
    n_n = len(y) - n_p
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(np.concatenate([scores[y > 0.5], scores[y < 0.5]]))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    rp = ranks[:n_p].sum()
    return float((rp - n_p * (n_p + 1) / 2) / (n_p * n_n))


def main():
    ckpt = os.environ.get("MED_CKPT", "med_cc3_ep8.npz")
    ckpt_path = os.path.join(PROJ, "lens", "runs", "vl", ckpt)
    if not os.path.exists(ckpt_path):
        print(f"no checkpoint {ckpt_path}; exit 2")
        return 2

    net, meta = load_vision(ckpt_path, device="cpu")
    n_img = len(net.img_concepts)
    n_txt = len(net.txt_concepts)
    print(f"ship: {os.path.basename(ckpt_path)}  img concepts {n_img}  text concepts {n_txt}")

    import vl_data
    import vl_data3
    # datum seed PINNED to 0: the measuring instrument must not follow the model
    base = vl_data.prepare_vqa("pathmnist", max_train=None, seed=0,
                               color=True)
    if base is None:
        print("pathmnist data load failed; exit 2")
        return 2
    d = vl_data3.prepare(base, "medical", seed=0)
    X, Q = d["Xte"], d["Qte"]
    if Q.shape[1] != meta.get("L"):
        print(f"qlen drift: datum {Q.shape[1]} vs ship {meta.get('L')}; refusing to guess")
        return 2
    if int(np.max(Q)) >= meta.get("QV", 1 << 30):
        print(f"question id {int(np.max(Q))} exceeds ship QV {meta.get('QV')}; refusing to guess")
        return 2
    n_use = 2000
    Xb, Qb = np.asarray(X[:n_use], dtype=np.float32), np.asarray(Q[:n_use], dtype=np.int64)

    out = net.forward_arrays(Xb, Qb)
    h = np.asarray(B.to_cpu(out["concepts"]), dtype=np.float64)

    # formula incidence for every image concept, on the same datum
    inc = np.zeros((n_use, n_img), dtype=np.float64)
    for k, c in enumerate(net.img_concepts):
        inc[:, k] = np.asarray(c.fn(Xb), dtype=np.float64)

    rows = []
    for k in range(n_img):
        own = corr(h[:, k], inc[:, k])
        best_other, best_j = 0.0, -1
        for j in range(n_img):
            if j == k:
                continue
            v = abs(corr(h[:, k], inc[:, j]))
            if v > best_other:
                best_other, best_j = v, j
        au = auroc(h[:, k], inc[:, k] > 0.5)
        margin = own - best_other
        if margin > 0.10:
            state = "honest"
        elif margin > -0.10:
            state = "weak"
        else:
            state = "mislabelled"
        rows.append((net.img_concepts[k].name, own, best_other,
                     net.img_concepts[best_j].name if best_j >= 0 else "-", au, margin, state))

    n_honest = sum(1 for r in rows if r[6] == "honest")
    n_weak = sum(1 for r in rows if r[6] == "weak")
    n_mis = sum(1 for r in rows if r[6] == "mislabelled")
    n_audef = sum(1 for r in rows if r[4] is not None and not np.isnan(r[4]) and r[4] < 0.8)

    print("\n  STRICT LABEL-HONESTY (test windows, |own| vs |best-other| margin > 0.10):")
    for r in sorted(rows, key=lambda r: abs(r[5])):
        print(f"    {r[0]:<26} {r[1]:7.3f} {r[2]:8.3f}  {r[3]:<16}{r[4]:7.3f} {r[5]:8.3f}  {r[6]}")
    print(f"\n  STRICT: honest {n_honest}/{n_img} (100% bar) "
          f"{'PASS' if n_honest == n_img else 'FAIL'}; "
          f"mislabelled {n_mis}; weak {n_weak}; AUROC<0.8: {n_audef}")

    out_path = os.path.join(PROJ, "lens", "runs", "vl", "med_strict_audit.json")
    json.dump({"ckpt": os.path.basename(ckpt_path), "n_img": n_img, "n_honest": n_honest,
               "n_weak": n_weak, "n_mislabelled": n_mis, "n_auroc_def": n_audef,
               "rows": [dict(zip(["name", "own", "best_other", "rival", "auroc", "margin", "state"],
                                 r)) for r in rows]},
              open(out_path, "w"), indent=1)
    print(f"  written {out_path}")
    return 0 if n_honest == n_img else 2


if __name__ == "__main__":
    sys.exit(main())