"""g2_audit_verify.py — CONTROL: does the G2 audit reproduce a KNOWN result?

Target E's readable model was audited by the project's own meter and scored **R1 7/9**. If my
rewritten audit does not reproduce that on E's checkpoint, my numbers mean nothing and G2 must
not be reported from them.

This is the check I should have run before the first G2 sweep. The first version of the audit used
|h * W| for contribution instead of the meter's logit-gain formula and reported 0/9 at D=32.
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np  # noqa: E402

from glassnet import tiny_data                     # noqa: E402
from glassnet.tensor import set_backend            # noqa: E402
from run_t2_iter003 import WIN, TinyLM3            # noqa: E402

set_backend("numpy")
R = os.path.join(PROJ, "runs", "t3")

DATA = tiny_data.build(win=WIN, vocab_cap=1500)
Xva = np.asarray(DATA["Xva"])
V = DATA["V"]
TARGETS = json.load(open(os.path.join(R, "meaningful_targets_ext.json")))["qualifying"]
AUROC_BAR, CONTRIB_BAR, FIRING_FLOOR = 0.55, 0.25, 0.02


def auroc(score, t01):
    o = np.argsort(score)
    r = np.empty(len(score), dtype=float)
    r[o] = np.arange(1, len(score) + 1)
    n1, n0 = int(t01.sum()), int((~t01).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[t01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def audit(net):
    out = net.forward(Xva)
    h = np.asarray(out["concepts"].data)
    L = np.log1p(np.asarray([(x != V - 1).sum() for x in Xva]))
    A = np.vstack([np.ones_like(L), L]).T
    rows = []
    for c, t in enumerate(TARGETS):
        ids = [i for i in t["ids"] if 0 <= i < V]
        firing = float((h[:, c] > 0.5).mean())
        contains = np.stack([np.isin(Xva, i).any(1) for i in ids]).any(0)
        coef, *_ = np.linalg.lstsq(A, h[:, c], rcond=None)
        resid = h[:, c] - A @ coef
        a = auroc(resid, contains)
        fired = h[:, c] > 0.5
        contrib = 0.0
        if fired.any():
            bk = net.concept.W.data[:, c].reshape(WIN, net.D)
            tot = 0.0
            for i in ids:
                wvec = bk @ net.emb.data[i]
                take = fired & np.isin(Xva, i).any(1)
                if take.any():
                    tot += float((wvec[None, :] * (Xva[take] == i)).sum())
            contrib = tot / float(fired.sum())
        ok = (a >= AUROC_BAR) and (contrib >= CONTRIB_BAR) and (firing >= FIRING_FLOOR)
        rows.append({"col": c, "name": t["name"], "auroc": float(a), "contrib": float(contrib),
                     "firing": firing, "pass": bool(ok)})
    return rows


print("=" * 100)
print("CONTROL — does the G2 audit reproduce Target E's known 7/9?")
print("=" * 100)

D = 64
net = TinyLM3(V, 9, D, WIN, rng=np.random.default_rng(0))
p = os.path.join(R, "ckpt", "readable_K9_lr1.0_lc01_meaningful_ext_e_seed0.npz")
z = np.load(p)
net.emb.data = z["emb"]
net.concept.W.data = z["concept.W"]
net.concept.b.data = z["concept.b"]
net.answer.W.data = z["answer.W"]
net.answer.b.data = z["answer.b"]
print(f"loaded {os.path.basename(p)} into a D={D} K=9 net")

rows = audit(net)
npass = sum(1 for r in rows if r["pass"])
print(f"\n{'col':>3} {'name':<24} {'auroc':>7} {'contrib':>9} {'firing':>7}  pass   (recorded E)")
recorded = {"pronoun_subject": ("0.5125", "0.0769"), "is_a_dialogue_line": ("0.5460", "0.0770"),
            "is_archaic_address": ("0.7485", "0.4147")}
for r in rows:
    rec = recorded.get(r["name"])
    tag = f"   <- recorded {rec[0]}/{rec[1]}" if rec else ""
    print(f"{r['col']:>3} {r['name']:<24} {r['auroc']:>7.4f} {r['contrib']:>9.4f} {r['firing']:>7.4f}  "
          f"{'PASS' if r['pass'] else 'FAIL'}{tag}")

print(f"\nMY AUDIT: R1 {npass}/9")
print("RECORDED: R1 7/9  (TARGET-E-RESULTS.md)")
print(f"\nCONTROL: {'PASS -- the audit matches the project meter' if npass == 7 else 'FAIL -- DO NOT REPORT G2 FROM THIS AUDIT'}")
