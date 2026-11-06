#!/usr/bin/env python3
"""TRAIN_SPEECH — a READABLE spoken-word recognizer on real audio (Speech Commands, CC BY 4.0).

Structure (same discipline as every other line: nothing hidden, every middle neuron has a name):
    features (9 measured formulas, see speech_data.py)
      -> concept layer   : 9 NAMED concepts, each trained to match its own measurable label
                           (long_sound, loud, noisy, bright, low_band_dominant, high_band_present,
                            voiced, tonal, early_peak)
      -> rule layer      : sparse rules (L1 penalty + top-k pruning), each rule reads few concepts
      -> answer layer    : the 8 words

Honest reporting rules obeyed here:
  * standardisation stats, concept thresholds and the vocabulary all come from the TRAIN split only
  * the official validation_list.txt examples are the held-out set; the majority-class baseline is
    printed beside the accuracy
  * a strict concept audit reports, per concept, whether its own label correlation BEATS its rivals
    (a concept that does not beat its rivals is reported as not honest, never papered over)
  * explain() prints the rules and concepts behind one prediction, with their numbers
Writes lens/runs/speech/speech_s0_ep*.npz and speech_report.json.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
import lens  # noqa: F401  (keeps the import surface identical to other trainers)
from lens import backend
import lens.tensor as T

DATA = os.path.join(PROJ, "data/speech/speech_v1.npz")
RUNS = os.path.join(PROJ, "lens/runs/speech")
CONCEPTS = ["long_sound", "loud", "noisy", "bright", "low_band_dominant", "high_band_present",
            "voiced", "tonal", "early_peak"]
# each concept = a threshold on ONE named feature (so its label is computable by hand)
CONCEPT_FEATURE = {"long_sound": "dur_s", "loud": "rms", "noisy": "zcr", "bright": "centroid_hz",
                   "low_band_dominant": "low_ratio", "high_band_present": "high_ratio",
                   "voiced": "voiced_frac", "tonal": "flatness", "early_peak": "peak_pos"}
CONCEPT_DIRECTION = {"tonal": "<", "early_peak": "<"}   # all others: feature > median


def auc(scores, labels):
    """Rank-based AUROC (no sklearn)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def main():
    os.makedirs(RUNS, exist_ok=True)
    d = np.load(DATA, allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split"]
    meta = json.load(open(os.path.join(PROJ, "data/speech/speech_v1_meta.json")))
    feats = list(meta["features"])          # names + formulas live in the meta file (documented)
    words = meta["words"]
    tr, va = split == 0, split == 1
    print(f"  clips {len(X):,}  train {tr.sum():,}  official-val {va.sum():,}  "
          f"words {words}  features {feats}")

    # --- DROP constant features and degenerate concepts (measured, printed, never forced) -----
    sd_all = X[tr].std(axis=0)
    drop_feat = [f for f, sdv in zip(feats, sd_all) if sdv < 1e-9]
    if drop_feat:
        print(f"  DROPPED constant features (no information, measured on the train split): {drop_feat}"
              f"  -- every Speech Commands clip is exactly 1.00 s, so dur_s cannot separate anything")
    keep_feats = [f for f in feats if f not in drop_feat]
    feats = keep_feats
    drop_con = [c for c in CONCEPTS if CONCEPT_FEATURE[c] in drop_feat]
    CONCEPTS[:] = [c for c in CONCEPTS if c not in drop_con]
    if drop_con:
        print(f"  DROPPED concepts whose only feature is constant: {drop_con}")

    # --- TRAIN-ONLY normalisation, concept labels, and honest concept dropping ----------------
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8
    Z = ((X - mu) / sd).astype(np.float32)
    thr = {c: float(np.median(X[tr][:, feats.index(CONCEPT_FEATURE[c])])) for c in CONCEPTS}
    labels = np.zeros((len(X), len(CONCEPTS)), dtype=np.float32)
    for k, c in enumerate(CONCEPTS):
        col = X[:, feats.index(CONCEPT_FEATURE[c])]
        labels[:, k] = (col < thr[c]) if CONCEPT_DIRECTION.get(c) == "<" else (col > thr[c])
    # degenerate concepts: a label that is (almost) always 0 or always 1 cannot be learned or
    # audited honestly -> dropped, with the measured rate printed (never forced, never hidden)
    rate = labels[tr].mean(axis=0)
    keep_k = [k for k in range(len(CONCEPTS)) if 0.01 <= rate[k] <= 0.99]
    if len(keep_k) < len(CONCEPTS):
        print("  DROPPED degenerate concepts (label constant on the TRAIN split):")
        for k in range(len(CONCEPTS)):
            if k not in keep_k:
                print(f"    {CONCEPTS[k]:<18} positive rate {rate[k]*100:.2f}%")
    CONCEPTS[:] = [CONCEPTS[k] for k in keep_k]
    C = labels[:, keep_k]
    print("  concept thresholds (medians of TRAIN split):")
    for c in CONCEPTS:
        dirn = "<" if CONCEPT_DIRECTION.get(c) == "<" else ">"
        print(f"    {c:<18} {CONCEPT_FEATURE[c]:<12} {dirn} {thr[c]:.4f}   "
              f"train-positive rate {C[tr, CONCEPTS.index(c)].mean()*100:.1f}%")

    # ABLATION HOOK: LENS_SPEECH_DROP=noisy re-trains without the named concepts. Used to test
    # whether the one dishonest concept (`noisy`, zero-crossing rate) is load-bearing or dead wood.
    _drop = [c for c in os.environ.get("LENS_SPEECH_DROP", "").split(",") if c]
    if _drop:
        keep_c = [c for c in CONCEPTS if c not in _drop]
        print(f"  ABLATION: dropping concepts {_drop} -> keeping {keep_c}")
        CONCEPTS[:] = keep_c
        C = C[:, [i for i, c in enumerate(CONCEPTS + _drop) if c in CONCEPTS][:len(keep_c)]]
    rng = np.random.default_rng(0)
    F, K, R, NCLS = len(feats), len(CONCEPTS), 24, len(words)
    W1 = T.Tensor(backend.to_dev((rng.standard_normal((F, K)) * (2 / F) ** 0.5).astype(np.float32)),
                  requires_grad=True)
    b1 = T.Tensor(backend.to_dev(np.zeros(K, dtype=np.float32)), requires_grad=True)
    W2 = T.Tensor(backend.to_dev((rng.standard_normal((K, R)) * (2 / K) ** 0.5).astype(np.float32)),
                  requires_grad=True)
    b2 = T.Tensor(backend.to_dev(np.zeros(R, dtype=np.float32)), requires_grad=True)
    W3 = T.Tensor(backend.to_dev((rng.standard_normal((R, NCLS)) * (2 / R) ** 0.5).astype(np.float32)),
                  requires_grad=True)
    b3 = T.Tensor(backend.to_dev(np.zeros(NCLS, dtype=np.float32)), requires_grad=True)
    params = [W1, b1, W2, b2, W3, b3]

    Zt = backend.to_dev(Z)
    Ct = backend.to_dev(C)
    onehot = np.zeros_like(C)
    onehot[np.arange(len(C)), np.argmax(C, axis=1)] = 1
    lambda_c, lambda_l1 = 0.5, 0.01
    lr0, epochs, batch = 0.3, 60, 256
    n_tr = int(tr.sum())
    idx_tr = np.where(tr)[0]
    hist = []
    for ep in range(epochs):
        lr = lr0 * (0.5 ** (ep / 30))
        order = rng.permutation(idx_tr)
        tot = 0.0
        for s in range(0, len(order), batch):
            ids = order[s:s + batch]
            h = T.sigmoid(T.add(T.matmul(T.Tensor(Zt[ids]), W1), b1))   # concept activations
            r = T.relu(T.add(T.matmul(h, W2), b2))                       # sparse rules
            logits = T.add(T.matmul(r, W3), b3)
            ce = T.softmax_ce(logits, y[ids])
            # concept loss: MSE between each concept activation and its own measured 0/1 label.
            # MSE (not BCE) because the engine's op set is deliberately small -- add/sub/mul/
            # matmul/relu/sigmoid/sum/softmax_ce -- and MSE is expressible with exactly those and
            # pushes the neuron the same way BCE would. Stated, not hidden.
            diff = T.sub(h, T.Tensor(Ct[ids]))
            cone = T.mul(T.sum_all(T.mul(diff, diff)),
                         T.Tensor(np.asarray(1.0 / (len(ids) * K), dtype=np.float32)))
            l1 = T.mul(T.sum_all(T.relu(W2)), T.Tensor(np.asarray(lambda_l1, dtype=np.float32)))
            loss = T.add(T.add(ce, T.mul(cone, T.Tensor(np.asarray(lambda_c, dtype=np.float32)))), l1)
            for p in params:
                p.grad = None
            loss.backward()
            for p in params:
                if p.grad is not None:
                    p.data = p.data - lr * p.grad
            tot += float(ce.data) * len(ids)
        # eval on the official val split (answer accuracy) every 10 epochs
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            Zv = backend.to_dev(Z[va])
            h = np.asarray(T.sigmoid(T.add(T.matmul(T.Tensor(Zv), W1), b1)).data)
            r = np.asarray(T.relu(T.add(T.matmul(T.Tensor(h), W2), b2)).data)
            lg = np.asarray(T.add(T.matmul(T.Tensor(r), W3), b3).data)
            acc = float((lg.argmax(axis=1) == y[va]).mean())
            hist.append({"epoch": ep + 1, "train_ce": tot / max(n_tr, 1), "val_acc": acc})
            print(f"  epoch {ep+1:>3}  train CE {tot/max(n_tr,1):.4f}   official-val acc "
                  f"{acc*100:.2f}%   (majority baseline {max(np.bincount(y[tr]))/n_tr*100:.2f}%)")
            np.savez_compressed(os.path.join(RUNS, f"speech_s0_ep{ep+1}.npz"),
                                W1=np.asarray(W1.data), b1=np.asarray(b1.data),
                                W2=np.asarray(W2.data), b2=np.asarray(b2.data),
                                W3=np.asarray(W3.data), b3=np.asarray(b3.data),
                                concepts=np.asarray(CONCEPTS), words=np.asarray(words),
                                feats=np.asarray(feats))

    # --- PRUNE the rule layer to top-3 concepts per rule (readability), then re-fit the answer ---
    W2n = np.asarray(W2.data)
    keep = np.zeros_like(W2n, dtype=bool)
    topk = np.argsort(-np.abs(W2n), axis=0)[:3, :]
    for rr in range(R):
        keep[topk[:, rr], rr] = True
    W2.data = backend.to_dev((W2n * keep).astype(np.float32))
    acc_before = hist[-1]["val_acc"]
    for ep in range(20):
        lr = 0.1 * (0.5 ** (ep / 10))
        order = rng.permutation(idx_tr)
        for s in range(0, len(order), batch):
            ids = order[s:s + batch]
            h = T.sigmoid(T.add(T.matmul(T.Tensor(Zt[ids]), W1), b1))
            r = T.relu(T.add(T.matmul(h, W2), b2))
            logits = T.add(T.matmul(r, W3), b3)
            loss = T.softmax_ce(logits, y[ids])
            for p in (W2, b2, W3, b3):
                p.grad = None
            loss.backward()
            for p in (W2, b2, W3, b3):
                if p.grad is not None:
                    p.data = p.data - lr * p.grad
            # ENFORCE the pruning mask every step. Without this the pruned weights grow back
            # during the re-fit and the reported "avg connections per rule" is a false claim
            # (measured: 4.88/rule instead of the 3.00 the top-3 pruning implies).
            W2.data = backend.to_dev((np.asarray(W2.data) * keep).astype(np.float32))
    Zv = backend.to_dev(Z[va])
    h_v = np.asarray(T.sigmoid(T.add(T.matmul(T.Tensor(Zv), W1), b1)).data)
    r_v = np.asarray(T.relu(T.add(T.matmul(T.Tensor(h_v), W2), b2)).data)
    lg_v = np.asarray(T.add(T.matmul(T.Tensor(r_v), W3), b3).data)
    acc_pruned = float((lg_v.argmax(axis=1) == y[va]).mean())
    conns = int((np.asarray(W2.data) != 0).sum())
    print(f"\n  after pruning to top-3 concepts/rule: val acc {acc_pruned*100:.2f}% "
          f"(was {acc_before*100:.2f}%)  connections {conns} "
          f"(avg {conns/R:.2f} per rule; dense would be {K}R)")

    # --- STRICT CONCEPT AUDIT: does each concept's activation track its OWN label best? --------
    Ztr = backend.to_dev(Z[tr])
    h_tr = np.asarray(T.sigmoid(T.add(T.matmul(T.Tensor(Ztr), W1), b1)).data)
    report = {"val_acc": acc_pruned, "val_acc_before_prune": acc_before,
              "majority_baseline": float(max(np.bincount(y[tr])) / n_tr), "n_val": int(va.sum()),
              "connections_per_rule": conns / R, "concepts": {}}
    print("\n  STRICT CONCEPT AUDIT (own label vs rivals):")
    honest = 0
    for k, c in enumerate(CONCEPTS):
        act = h_tr[:, k]
        cors = {c2: float(np.corrcoef(act, C[tr, j])[0, 1]) for j, c2 in enumerate(CONCEPTS)}
        own = cors[c]
        rival_name, rival_val = max(((n2, v) for n2, v in cors.items() if n2 != c),
                                   key=lambda kv: kv[1])
        a = auc(act, C[tr, k])
        ok = own > rival_val
        honest += int(ok)
        report["concepts"][c] = {"own_corr": own, "best_rival": rival_name,
                                "best_rival_corr": rival_val, "auroc": a, "honest": ok}
        print(f"    {c:<18} own {own:+.3f}  best rival {rival_name} {rival_val:+.3f}  "
              f"AUROC {a:.3f}   {'honest' if ok else 'NOT HONEST'}")
    report["honest_concepts"] = f"{honest}/{len(CONCEPTS)}"

    # --- explain(): the arithmetic behind one test prediction ----------------------------------
    j = int(np.where(va)[0][0])
    r_one = r_v[0]
    top_rules = np.argsort(-r_one)[:3]
    pred = int(lg_v[0].argmax())
    print(f"\n  explain() sample: predicted '{words[pred]}' (true '{words[y[j]]}')")
    for rr in top_rules:
        cons = [(CONCEPTS[k], float(np.asarray(W2.data)[k, rr])) for k in np.argsort(-np.abs(np.asarray(W2.data)[:, rr]))[:3] if np.asarray(W2.data)[k, rr] != 0]
        cons_s = ", ".join(f"{n} (w={w:+.2f})" for n, w in cons)
        print(f"    rule #{rr} (activation {r_one[rr]:.3f})  reads: {cons_s}")
    print(f"    answer '{words[pred]}' score {float(lg_v[0][pred]):.4f} vs runner-up "
          f"{float(np.sort(lg_v[0])[-2]):.4f}")
    # SHIP the PRUNED model: the readable artifact is the one with <=3 concepts per rule, NOT the
    # dense mid-training checkpoint. (selfcheck caught this: it was checking the ep60 checkpoint and
    # correctly reported 8.00 connections/rule, i.e. the shippable artifact was missing.)
    tag = ("_drop-" + "-".join(_drop)) if _drop else ""
    ship_path = os.path.join(RUNS, f"speech_readable_s0{tag}.npz")
    np.savez_compressed(ship_path,
                        W1=np.asarray(W1.data), b1=np.asarray(b1.data),
                        W2=np.asarray(W2.data), b2=np.asarray(b2.data),
                        W3=np.asarray(W3.data), b3=np.asarray(b3.data),
                        concepts=np.asarray(CONCEPTS), words=np.asarray(words),
                        feats=np.asarray(feats), pruned=np.asarray(True),
                        mu=mu.astype(np.float32), sd=sd.astype(np.float32),
                        val_acc=np.asarray(acc_pruned), connections_per_rule=np.asarray(conns / R))
    print(f"  shipped pruned model: {ship_path}  ({conns/R:.2f} connections/rule, "
          f"val acc {acc_pruned*100:.2f}%)")
    json.dump({**report, "history": hist, "words": words, "features": feats,
               "concept_formulas": meta["formulas"], "concept_feature": CONCEPT_FEATURE,
               "thresholds": thr,
               "data": {"n": int(len(X)), "val": "official validation_list/testing_list",
                        "license": "CC BY 4.0", "dup_rows_across_split":
                        meta.get("dup_rows_across_split")}},
              open(os.path.join(RUNS, f"speech_report{tag}.json"), "w"), indent=1, default=str)
    print(f"\n  report: {RUNS}/speech_report{tag}.json")


if __name__ == "__main__":
    main()