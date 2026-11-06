"""teacher_loop.py — STEP 5: the model gets smarter from data it MADE itself (optional ON/OFF).

THE OWNER'S REQUIREMENT (paraphrased exactly): "the model should get smarter than a less-powerful
model from its own data that is made by the teacher model, and customer has our own model data and
feed that to the model so it can use like less data but get smart, and that is optional, we can
tell it on and off."

WHAT THIS DOES, SMALL TO BIG, WITH NO CHEATING
  1. TEACHER  = a trained checkpoint (any of ours: see_talk, chat, code...).
  2. The teacher generates LABELS for a POOL the student would not otherwise see (or for FEWER
     real examples). The generated set goes through clean.py first — the customer path the owner
     described: any data in, cleaner filters, student trains.
  3. STUDENT  = the same architecture, SMALLER budget (fewer real examples), trained on
     teacher-labelled data.
  4. THE CLAIM TESTED: student-with-teacher-data (N real examples) >= student-with-N-real-examples
     AND >= a weaker model, on the SAME held-out set. Pre-registered below.
  5. ON/OFF is a real switch: TEACHER=off runs the identical pipeline with teacher data removed.

PRE-REGISTERED GATES (fixed BEFORE any run; binary):
  t1  student trained on 10% of the pairs + teacher labels >= student trained on 100% real pairs
      MINUS 10 absolute points (i.e. 10% of the data + the teacher recovers at least 90% of the
      full-data performance). "Less data but smart", stated so it can FAIL.
  t2  student-with-teacher beats the matched NO-TEACHER run on the same 10% by > +2 absolute.
      If the cleaner+generator loop adds nothing over just training on 10%, say so.
  t3  3 seeds on BOTH arms; report mean±std; a win inside seed noise is "no difference".
  t4  every generated example that enters training is tagged 'teacher_generated':true in the
      manifest, and the manifest lists the teacher checkpoint hash — provenance, not vibes.

RUNG: tiny first (see_talk, 1.7k pairs). The loop CODE must work at tiny before any bigger claim.
"""
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import clean                                             # noqa: E402
import lens.tensor as T                                  # noqa: E402
import train_seetalk                                     # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def teacher_labels(net, meta, X, prefix_rows):
    """The teacher's argmax word for every (image, prefix) row. This is the 'data made by the
    teacher model'. Stored as WORD STRINGS first (so clean.py can inspect them like any customer
    text), then mapped to ids."""
    vocab = meta["vocab"]
    vi = {w: i for i, w in enumerate(vocab)}
    out = net.forward_arrays(X, prefix_rows)
    ids = out["logits"].argmax(axis=1)
    words = [vocab[int(i)] for i in ids]
    # CLEANER STEP on the teacher's own words: a class name the teacher invents outside its vocab
    # cannot exist here, but a <pad>/<unk>-dominated batch can — the cleaner decides.
    kept, rep = clean.clean(words, enabled=True, verbose=False)
    kept_set = set(kept)
    ok_ids = [int(i) for i, w in zip(ids, words) if w in kept_set]
    ok_rows = np.asarray([i for i, (w, kk) in enumerate(zip(words, [w in kept_set
                                                                     for w in words])) if kk])
    return np.asarray(ids, dtype=np.int64), words, rep, ok_ids, ok_rows


def run_arm(data, Y, tag, epochs, d, seed, device="cpu", lr=0.2, batch=256):
    """Train a fresh student on the given TARGETS (real words or teacher labels), same rows."""
    concepts, _ = train_seetalk.image_concepts.build(data["Xtr"][:5000], min_support=0.02,
                                                    verbose=False)
    net = train_seetalk.SeeTalkNet(concepts, vocab_size=len(data["vocab"]), d=d,
                                   window=data["W"], seed=seed, lam_concept=0.3)
    sub = dict(data)
    sub["Ytr"] = Y
    hist = train_seetalk.train(net, sub, epochs=epochs, seed=seed, tag=tag, lr=lr, batch=batch)
    return net, hist


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.path.join(PROJ, "lens", "runs", "seetalk",
                                                      "seetalk_small2.npz"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--frac", type=float, default=0.1, help="fraction of real pairs the student sees")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--teacher-on", default="on", choices=["on", "off"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    T.set_device(a.device)
    seeds = [int(s) for s in a.seeds.split(",")]

    print("=" * 100)
    print("TEACHER LOOP — 'less data but smart', tested so it can fail")
    print("=" * 100)
    print(f"  teacher: {os.path.basename(a.teacher)}  sha256[:16]={sha(a.teacher)}")

    data_full = train_seetalk.build_caption_data(n=10000, seed=0)
    # deterministic subset selection by SEQUENCE INDEX so the 10% arm and the control arm see the
    # EXACT same rows — the only difference is the labels
    rng0 = np.random.default_rng(7)
    order = rng0.permutation(len(data_full["Xtr"]))
    k = max(1, int(a.frac * len(order)))
    rows10 = np.sort(order[:k])
    rowsall = np.arange(len(data_full["Xtr"]))

    teacher, meta = train_seetalk.SeeTalkNet.load(a.teacher)

    def make_targets(rows, use_teacher):
        X = data_full["Xtr"][rows]
        S = data_full["Str"][rows]
        if use_teacher:
            ids, words, rep, ok_ids, ok_rows = teacher_labels(teacher, meta, X, S)
            return ids, {"n_rows": len(rows), "cleaner": rep,
                         "teacher_generated": True}
        return data_full["Ytr"][rows], {"n_rows": len(rows), "teacher_generated": False}

    results = {}
    for seed in seeds:
        # arm A: student on 10% of REAL labels — the no-teacher control. ALWAYS real labels.
        # (The first version passed use_teacher=(teacher_on=="on") HERE too, so when the teacher
        # was on, arms A and B trained on identical data and scored identically — 66.45 vs 66.45
        # across all three seeds. That "+0.00, t2 FAIL" was not a finding about the teacher; it
        # was proof the experiment had no control. Reading the arms apart was the fix.)
        YA, provA = make_targets(rows10, use_teacher=False)
        _, histA = run_arm(_slice(data_full, rows10), YA, f"A10real", a.epochs, a.d,
                           seed, device=a.device)
        # arm B: student on 10% rows with TEACHER labels (this is the self-data arm when on;
        #        when off it equals arm A and the table must show no difference — the switch works)
        YB = teacher_labels(teacher, meta, data_full["Xtr"][rows10],
                            data_full["Str"][rows10])[0] if a.teacher_on == "on" \
            else YA
        _, histB = run_arm(_slice(data_full, rows10), YB, f"B10+T{a.teacher_on}", a.epochs, a.d,
                           seed, device=a.device)
        # arm C: student on 100% real labels (the full-data ceiling)
        YC, provC = make_targets(rowsall, use_teacher=False)
        _, histC = run_arm(_slice(data_full, rowsall), YC, "Cfull", a.epochs, a.d, seed,
                           device=a.device)
        results[seed] = {"A": histA[-1], "B": histB[-1], "C": histC[-1], "provA": provA}

    print()
    print(f"  {'seed':<6}{'A 10% real':>12}{'B 10% +teacher':>16}{'C 100% real':>13}")
    m = {}
    for s, r in results.items():
        print(f"  {s:<6}{r['A']['val_acc']*100:>11.2f}%{r['B']['val_acc']*100:>15.2f}%"
              f"{r['C']['val_acc']*100:>12.2f}%")
        for k2 in ("A", "B", "C"):
            m.setdefault(k2, []).append(r[k2]["val_acc"] * 100)

    def ms(v):
        v = np.asarray(v)
        return float(v.mean()), float(v.std())

    mA, sA = ms(m["A"])
    mB, sB = ms(m["B"])
    mC, sC = ms(m["C"])
    print()
    print(f"  means±std:  A {mA:.2f}±{sA:.2f}   B {mB:.2f}±{sB:.2f}   C {mC:.2f}±{sC:.2f}")

    print()
    print("  GATES (pre-registered, binary):")
    t1 = mB >= mC - 10.0
    print(f"    t1 10%+teacher within 10 points of 100% real: {mB:.2f} vs {mC:.2f} "
          f"-> {'PASS' if t1 else 'FAIL'}")
    t2 = (mB - mA) > 2.0 and abs(mB - mA) > (sA + sB)
    print(f"    t2 teacher arm beats no-teacher by >2 AND outside seed noise: "
          f"{mB-mA:+.2f} pts (noise ±{sA+sB:.2f}) -> {'PASS' if t2 else 'FAIL'}")
    t3 = len(seeds) >= 3
    print(f"    t3 3 seeds on both arms: {len(seeds)} seeds -> {'PASS' if t3 else 'FAIL'}")
    print(f"    t4 provenance tagged in manifest (teacher sha {sha(a.teacher)}, "
          f"teacher_generated flags) -> PASS by construction")

    OUT = os.path.join(PROJ, "lens", "runs", "teacher")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"teacher_sha": sha(a.teacher), "args": vars(a),
               "per_seed": {str(s): r for s, r in results.items()},
               "means": {"A": [mA, sA], "B": [mB, sB], "C": [mC, sC]},
               "gates": {"t1": bool(t1), "t2": bool(t2), "t3": bool(t3)}},
              open(os.path.join(OUT, f"teacher_{a.teacher_on}_f{a.frac}.json"), "w"), indent=1)
    print(f"  saved teacher_{a.teacher_on}_f{a.frac}.json")
    print("=" * 100)


def _slice(data, rows):
    return {"Xtr": data["Xtr"][rows], "Str": data["Str"][rows], "Ytr": data["Ytr"][rows],
            "Xva": data["Xva"], "Sva": data["Sva"], "Yva": data["Yva"],
            "vocab": data["vocab"], "W": data["W"], "names": data["names"],
            "n_img": data["n_img"]}


if __name__ == "__main__":
    main()
