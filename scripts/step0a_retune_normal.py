"""STEP 0a — PathMNIST normal re-baseline (the 37.99% is suspicious).

The rung-2 normal was trained with the same cheap 12-epoch/lr-0.1 schedule
that suited Fashion; PathMNIST's 90k-row train set deserves a proper tune.
Procedure (per the direction): VALIDATION-only decisions — grid {epochs
30/60} x {lr 0.1/0.2} on seed 0, pick the best config, then 3 seeds at it,
same parameter budget as the readable model (101,641 vs 101,593). Reports
the corrected gap, whatever it is. Test use logged separately (rung-2
re-check, touch #4).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from glassnet.loop_lib import load_split, train_normal, match_hidden, DATA_DIR
from glassnet.tensor import Tensor

def _predict(net, X):
    return np.argmax(net(Tensor(X)).data, axis=1).astype(np.int64)

TASK = "pathmnist"
Xtr, ytr, Xval, yval, Xte, yte = load_split(TASK)
classes = int(np.max(ytr)) + 1
d_in = Xtr.shape[1]
readable_val = 33.94   # iter 014, 3-seed mean (the milestone candidate)
readable_test = 46.20  # logged touch #3
hidden = match_hidden(101641, classes=classes, d_in=d_in)
n_params = hidden * d_in + hidden * classes + hidden + classes
print(f"[step0a] {TASK}: hidden={hidden} params={n_params} "
      f"(readable 101,593) | train={Xtr.shape}")

grid = [(30, 0.1), (60, 0.1), (30, 0.2), (60, 0.2)]
rows = []
for ep, lr in grid:
    t0 = time.time()
    net, v = train_normal(Xtr, ytr, Xval, yval, seed=0, hidden=hidden,
                          epochs=ep, batch=128, lr=lr, classes=classes,
                          d_in=d_in)
    rows.append((v, ep, lr, time.time() - t0))
    print(f"  grid seed0 epochs={ep} lr={lr}: val={v*100:.2f}% "
          f"({time.time()-t0:.0f}s)")
rows.sort(reverse=True)
best_ep, best_lr = rows[0][1], rows[0][2]
print(f"[step0a] best config: epochs={best_ep} lr={best_lr} (val {rows[0][0]*100:.2f}%)")

os.makedirs(os.path.join(DATA_DIR, "..", "runs", "step0a"), exist_ok=True)
outdir = os.path.normpath(os.path.join(DATA_DIR, "..", "runs", "step0a"))
v_vals, t_vals = {}, {}
for s in (0, 1, 2):
    net, v = train_normal(Xtr, ytr, Xval, yval, seed=s, hidden=hidden,
                          epochs=best_ep, batch=128, lr=best_lr,
                          classes=classes, d_in=d_in)
    v_vals[s] = v * 100
    t_vals[s] = float(np.mean(_predict(net, Xte) == yte)) * 100
    net.save(os.path.join(outdir, f"normal_recheck_seed{s}.npz"))
vm, vs = np.mean(list(v_vals.values())), np.std(list(v_vals.values()))
tm, ts = np.mean(list(t_vals.values())), np.std(list(t_vals.values()))
print(f"\n[step0a] corrected normal VAL = {vm:.2f} ± {vs:.2f}  "
      f"| TEST = {tm:.2f} ± {ts:.2f}")
print(f"[step0a] corrected gaps (readable - normal):"
      f"\n   VAL: {readable_val:.2f} - {vm:.2f} = {readable_val - vm:+.2f} pts"
      f"\n   TEST: {readable_test:.2f} - {tm:.2f} = {readable_test - tm:+.2f} pts")
json.dump({"config": {"epochs": best_ep, "lr": best_lr, "hidden": hidden,
                      "params": n_params},
           "grid": [{"val": r[0], "epochs": r[1], "lr": r[2]} for r in rows],
           "val_vals": v_vals, "test_vals": t_vals,
           "val_mean": vm, "test_mean": tm},
          open(os.path.join(outdir, "step0a_result.json"), "w"), indent=1)
print("[step0a] saved", os.path.join(outdir, "step0a_result.json"))