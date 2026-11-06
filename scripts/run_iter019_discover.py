"""ITER 019 — STEP 1 first run: discovered concepts (learned + auto-named).

Hypothesis (written BEFORE the run):
  a concept layer of K=24 LEARNED sigmoid neurons with sparsity penalties
  (lam_act=1.0 -> neurons mostly OFF; lam_w=1.0 -> few words per neuron),
  classes reading the discovered concepts directly (no gate, no rules),
  auto-named from top-weight words, can (1) beat the hand-picked baseline
  (readable VAL 11.68, iter 017) and (2) yield names that pass the held-out
  audit (fires on sentences containing the named words, quiet on others).

Success test (all three, unchanged after seeing results):
  A) readable VAL mean >= 11.68
  B) >= 50% of auto-named neurons pass the held-out audit (AUROC >= 0.55)
  C) pinned lie probe (concept column frozen at init) is caught (flagged
     BROKEN by the same audit)
Verdict = KEEP only if A and B and C; else REVERT (with the failure written
out plainly). Test set UNTOUCHED this iteration.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from glassnet import text_data as TD
from glassnet.discover import (train_discover, auto_name, audit_discover,
                               discover_params)
from glassnet.loop_lib import (load_split, match_hidden, train_normal,
                               RUNS, _relu)
from glassnet.layers import Dense, Sequential

K, EPOCHS, LR, LAM_A, LAM_W = 24, 1536, 0.1, 0.2, 1.0
USE_BIGRAMS = os.environ.get("GLASSNET_BIGRAMS", "0") == "1"   # iter 027-028 experiment, reverted
SEEDS = (0, 1, 2)
CLASSES, D_IN = 77, 1175
BEST = 84.63                      # iter 032 readable VAL mean, new bar

Xtr, ytr, Xval, yval, Xte, yte = load_split("banking77")
_, _, _, _, _, _, t_tr, t_va, t_te, uniq = TD.load()
vocab = TD.load_vocab()

# ---- BIGRAM features (word order) — iter 027-028 experiment, opt-in -------
_bg = []
if USE_BIGRAMS:
    from collections import Counter
    _tr_toks = [TD.tokenize(t) for t in t_tr]
    _cnt = Counter()
    for toks in _tr_toks:
        for a, b in zip(toks, toks[1:]):
            _cnt[(a, b)] += 1
    _bg = [f"{a}_{b}" for (a, b), _ in _cnt.most_common(2048)]
    _bgmap = {p: i for i, p in enumerate(_bg)}
    def _bg_mat(toks_list):
        M = np.zeros((len(toks_list), len(_bg)), dtype=np.float32)
        for i, toks in enumerate(toks_list):
            for a, b in zip(toks, toks[1:]):
                j = _bgmap.get((a, b))
                if j is not None:
                    M[i, j] += 1.0
        return np.log1p(M)
    Xtr = np.hstack([Xtr, _bg_mat(_tr_toks)])
    Xval = np.hstack([Xval, _bg_mat([TD.tokenize(t) for t in t_va])])
D_IN = Xtr.shape[1]
print(f"[iter029] bigram features: +{len(_bg)} -> input dim {D_IN}")
assert vocab is not None and len(vocab) == 1175, f"vocab missing ({len(vocab) if vocab else None})"

iterdir = os.path.join(RUNS, "iter033")
os.makedirs(iterdir, exist_ok=True)
t0 = time.time()

# ---- 1. discovered concepts, 3 seeds --------------------------------
rvals, nets = {}, {}
for s in SEEDS:
    net, v = train_discover(Xtr, ytr, Xval, yval, K, CLASSES, seed=s,
                            epochs=EPOCHS, batch=128, lr=LR,
                            lam_act=LAM_A, lam_w=LAM_W)
    rvals[s] = v * 100
    nets[s] = net
    net.save(os.path.join(iterdir, f"final_seed{s}.npz"))
rm, rs = float(np.mean(list(rvals.values()))), float(np.std(list(rvals.values())))
print(f"[iter019] discovered concepts VAL: {rm:.2f} ± {rs:.2f}  vals={ {s: round(rvals[s],2) for s in rvals} }")

# ---- 2. same-size normal, same budget --------------------------------
r_params = discover_params(K, CLASSES, D_IN)
hidden = match_hidden(r_params, classes=CLASSES, d_in=D_IN)
n_params = hidden * D_IN + hidden * CLASSES + hidden + CLASSES
nvals = {}
for s in SEEDS:
    net, v = train_normal(Xtr, ytr, Xval, yval, seed=s, hidden=hidden,
                          epochs=EPOCHS, batch=128, lr=LR, classes=CLASSES,
                          d_in=D_IN)
    nvals[s] = v * 100
    net.save(os.path.join(iterdir, f"normal_seed{s}.npz"))
nm, ns = float(np.mean(list(nvals.values()))), float(np.std(list(nvals.values())))
print(f"[iter019] same-size normal VAL: {nm:.2f} ± {ns:.2f}  "
      f"(r-params {r_params} vs n-params {n_params})")

# ---- 3. auto-names + held-out audit (seed 0 net) ----------------------
FEAT_NAMES = vocab + (_bg if USE_BIGRAMS else [])
net0 = nets[0]
names, examples = auto_name(net0, FEAT_NAMES, Xtr, t_tr)
rows, n_pass = audit_discover(net0, names, Xval, t_va)
aucs = [r["auroc"] for r in rows if r["flag"] != "NO-NAME"]
print(f"[iter019] auto-named neurons: audit pass {n_pass}/{K} "
      f"(mean AUROC {np.nanmean(aucs):.3f})")
for r in rows:
    print(f"    n{r['neuron']:02d} {r['name'][:44]:44s} auroc={r['auroc'] if not np.isnan(r['auroc']) else float('nan'):.2f} "
          f"quiet={r['quiet_mean']:.3f} loud={r['loud_mean']:.3f} {r['flag']}")

# ---- 4. lie probe: frozen concept column ------------------------------
probe_k = 2   # arbitrary column; its weights stay at init by construction
net_p, _ = train_discover(Xtr, ytr, Xval, yval, K, CLASSES, seed=0,
                          epochs=EPOCHS, batch=128, lr=LR,
                          lam_act=LAM_A, lam_w=LAM_W, frozen=[probe_k])
pnames, _ = auto_name(net_p, FEAT_NAMES, Xtr, t_tr)
prows, _ = audit_discover(net_p, pnames, Xval, t_va)
lie = prows[probe_k]
lie_caught = lie["flag"] == "BROKEN" or (not np.isnan(lie["auroc"]) and lie["auroc"] < 0.55)
print(f"[iter019] LIE PROBE (frozen column {probe_k}, name '{pnames[probe_k]}'): "
      f"auroc={lie['auroc']:.2f} contrib={lie['contrib']:.3f} -> CAUGHT={lie_caught}")

# ---- 5. verdict (read-only, fixed success test) ------------------------
okA = bool(rm >= BEST)
okB = bool(n_pass / K >= 0.50)
okC = bool(lie_caught)
print(f"\n[iter019] success tests: A(>=11.68)={okA} | B(>=50% audit)={okB} "
      f"| C(lie caught)={okC}")
verdict = "KEEP" if (okA and okB and okC) else "REVERT"
print(f"[iter019] verdict: {verdict}  ({time.time()-t0:.0f}s)")

json.dump({"iteration": 33, "task": "banking77", "mode": "discovered",
           "K": K, "epochs": EPOCHS, "lr": LR, "lam_act": LAM_A,
           "lam_w": LAM_W, "change": "epochs 768 -> 1536 (keep the budget "
                                    "sweep; A bar 84.63)",
           "input_dim": D_IN,
           "rval_vals": {str(s): v for s, v in rvals.items()},
           "rval_mean": rm, "rval_std": rs,
           "nval_vals": {str(s): v for s, v in nvals.items()},
           "nval_mean": nm, "nval_std": ns,
           "readable_params": r_params, "normal_params": n_params,
           "audit_pass": n_pass, "audit_total": K,
           "lie_caught": bool(lie_caught), "okA": okA, "okB": okB,
           "okC": okC, "verdict": verdict,
           "names": names, "examples": examples},
          open(os.path.join(iterdir, "results.json"), "w"), indent=1)
print("saved", os.path.join(iterdir, "results.json"))