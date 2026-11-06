"""f_audit.py — do the concepts SURVIVE a generation of self-training?

Targets D and E both lost 2 of their concepts. Those were trained on real data. These students
were trained on their teacher's own generated output, which is a further distribution shift.

For each readable student checkpoint: readability coverage, R1 (Meter-03), and R5 (METER-05
category ablation, all nine declared CATEGORICAL in TARGET-F.md).
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np
from glassnet.tensor import set_backend
from glassnet import tiny_data
import t3_audit_m3 as M3

R = os.path.join(PROJ, "runs", "t3")
N_TAUGHT, N_CONTROL = 9, 20

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
VOCAB = DATA["vocab"]
V = DATA["V"]
set_backend("numpy")
tj = json.load(open(os.path.join(R, "meaningful_targets_ext.json")))["qualifying"]
names = [t["name"] for t in tj]
tgt = M3.load_targets("meaningful_ext")

def load_student(path):
    """Load a checkpoint saved by f_train.py.

    f_train saves SHORT keys (emb/Wc/bc/Wa/ba); M3.load_ckpt matches by parameter NAME
    (emb/concept.W/concept.b/answer.W/answer.b). Loading an f_train checkpoint with M3.load_ckpt
    silently kept only the embedding and left the concept and answer layers RANDOM -- which is
    why the first F audit reported AUROC 0.50 for every concept and I wrongly reported that all
    nine concepts had died. This loader maps the keys explicitly.
    """
    z = np.load(path)
    net = M3.build_net(0)
    net.emb.data = z["emb"]
    net.concept.W.data = z["Wc"]
    net.concept.b.data = z["bc"]
    net.answer.W.data = z["Wa"]
    net.answer.b.data = z["ba"]
    return net


print("=" * 104)
print("TARGET F — DO THE CONCEPTS SURVIVE SELF-TRAINING? (readable students, real validation)")
print("=" * 104)

out = {}
for rule in ("all", "conf50"):
    print(f"\n--- rule {rule}")
    print(f"{'col':>3} {'concept':<22} {'AUROC':>7} {'contrib':>9} {'firing':>8} {'R1':>6} "
          f"{'d_all':>8} {'R5':>6}")
    r1n = r5n = 0
    rows = []
    for seed in (0, 1, 2):
        f = os.path.join(R, "ckpt", f"student_readable_{rule}_f2_seed{seed}.npz")
        if not os.path.exists(f):
            print(f"  seed {seed}: MISSING {f}")
            continue
        net = load_student(f)
        h, L, A = M3.prep(net, X)
        h = np.asarray(h)
        logits = np.asarray(net.forward(X)["logits"].data)
        W = net.answer.W.data
        pred = logits.argmax(1)
        contrib = h * W[:, pred].T
        mass = np.abs(contrib)
        share = mass[:, :N_TAUGHT].sum(1) / np.maximum(mass.sum(1), 1e-12)

        emb = net.emb.data
        backup = emb.copy()
        rng = np.random.default_rng(0)
        seed_rows = []
        for c in range(N_TAUGHT):
            r = M3.audit_col(net, X, h, c, tgt[c][1], L, A)
            r1 = (r["auroc"] >= 0.55) and (r["contrib"] >= 0.25) and (r["firing"] >= 0.02)
            ids = [i for i in tj[c]["ids"] if 0 <= i < V]
            act = h[:, c]
            fired = act > 0.5
            A0 = float(act[fired].mean()) if fired.sum() else 0.0

            def drop_for(idset):
                for i in idset:
                    emb[i] = 0.0
                hw = np.asarray(net.forward(X)["concepts"].data)[:, c]
                for i in idset:
                    emb[i] = backup[i]
                return (A0 - float(hw[fired].mean())) / A0 if A0 > 0 else 0.0

            d_all = drop_for(ids)
            rand = [drop_for(rng.choice(V - 1, size=len(ids), replace=False))
                    for _ in range(N_CONTROL)]
            z = (d_all - float(np.mean(rand))) / (float(np.std(rand)) + 1e-9)
            r5 = (d_all >= 0.25) and (z >= 3.0)
            seed_rows.append({"col": c, "name": names[c], "auroc": float(r["auroc"]),
                              "contrib": float(r["contrib"]), "firing": float(r["firing"]),
                              "r1": bool(r1), "d_all": float(d_all), "z": float(z),
                              "r5": bool(r5)})
            if seed == 0:
                print(f"{c:>3} {names[c]:<22} {r['auroc']:>7.4f} {r['contrib']:>9.4f} "
                      f"{r['firing']:>8.4f} {'PASS' if r1 else 'FAIL':>6} {d_all:>8.4f} "
                      f"{'PASS' if r5 else 'FAIL':>6}")
        emb[:] = backup
        r1s = sum(x["r1"] for x in seed_rows)
        r5s = sum(x["r5"] for x in seed_rows)
        print(f"    seed {seed}: R1 {r1s}/9  R5 {r5s}/9  coverage {share.mean()*100:.4f}% "
              f"(min {share.min()*100:.4f}%)")
        rows.append({"seed": seed, "r1": r1s, "r5": r5s,
                     "coverage_mean": float(share.mean()), "coverage_min": float(share.min()),
                     "per_concept": seed_rows})
    out[rule] = rows

print("\n" + "=" * 104)
print("COMPARISON — concepts surviving, by generation")
print("=" * 104)
print(f"  {'model':<34} {'R1':>8} {'R5':>8} {'coverage':>12}")
print(f"  {'Target E (trained on REAL data)':<34} {'7/9':>8} {'6/9':>8} {'100.0000%':>12}")
for rule in ("all", "conf50"):
    if out[rule]:
        r1 = np.mean([r["r1"] for r in out[rule]])
        r5 = np.mean([r["r5"] for r in out[rule]])
        cov = np.mean([r["coverage_mean"] for r in out[rule]]) * 100
        print(f"  {'F student (' + rule + ', generated data)':<34} {r1:>8.1f} {r5:>8.1f} "
              f"{cov:>11.4f}%")
json.dump(out, open(os.path.join(R, "f_audit.json"), "w"), indent=1)
print("\nsaved runs/t3/f_audit.json")
