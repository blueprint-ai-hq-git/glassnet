"""milestone_check.py — Rung-1 gate on the quarantined TEST set (one touch).

Loads the promoted iteration's saved models (readable + same-size normal),
evaluates BOTH on the official test split exactly once, logs the use in
TEST-USE.md, and checks the gate:
  |readable - normal| <= 2.0 pts  AND  readable >= normal  (means over 3 seeds)
If passed: writes STOPPED.md and exits 42 ("stop and wait for Yahya").

Run:  python3 milestone_check.py --iter 9
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.loop_lib import (HERE, RUNS, load_split,  # noqa: E402
                               normal_params, match_hidden, log_test_use,
                               readable_params)
from glassnet.phase2_model import GlassNetPhase2  # noqa: E402
from glassnet.layers import Dense, Sequential  # noqa: E402
from glassnet.tensor import relu  # noqa: E402
from glassnet.eval import accuracy as acc_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--task", default="fashion-mnist",
                    choices=["fashion-mnist", "pathmnist", "banking77"])
    ap.add_argument("--label", default="RUNG-GATE",
                    help="milestone label, e.g. rung-1 or rung-2")
    args = ap.parse_args()
    iterdir = os.path.join(RUNS, f"iter{args.iter:03d}")
    with open(os.path.join(iterdir, "results.json")) as f:
        res = json.load(f)

    Xtr, ytr, Xval, yval, Xte, yte = load_split(args.task)
    n_c = res.get("classes") or (77 if args.task == "banking77" else 10)
    d_in = res.get("input_dim", 784)

    # readable: 3 seeds, load from archive (discovered / phase2 auto-read)
    r_test = []
    if res.get("mode") == "discovered":
        from glassnet.discover import DiscoverNet
        from glassnet.tensor import Tensor
        for s in (0, 1, 2):
            m = DiscoverNet.load(os.path.join(iterdir, f"final_seed{s}.npz"))
            pred = m.forward(Tensor(Xte))["logits"].data.argmax(1)
            r_test.append(float(np.mean(pred == np.asarray(yte))))
    else:
        for s in (0, 1, 2):
            m = GlassNetPhase2.load(os.path.join(iterdir, f"final_seed{s}.npz"))
            r_test.append(acc_fn(m, Xte, yte))

    # same-size normal: reconstruct at the recorded hidden size
    hidden = res.get("normal_hidden") or match_hidden(
        res["readable_params"], classes=n_c, d_in=d_in)
    n_test = []
    for s in (0, 1, 2):
        net = Sequential(Dense(d_in, hidden, rng=np.random.default_rng(s),
                               name="n1"), relu,
                         Dense(hidden, n_c, rng=np.random.default_rng(s),
                               name="n2"))
        z = np.load(os.path.join(iterdir, f"normal_seed{s}.npz"))
        for layer in net.layers:
            if hasattr(layer, "parameters"):
                for p in layer.parameters():
                    p.data[:] = z[p.name].astype(np.float32)
        n_test.append(acc_fn(net, Xte, yte))

    trm, trs = float(np.mean(r_test)), float(np.std(r_test))
    tnm, tns = float(np.mean(n_test)), float(np.std(n_test))
    rp = res["readable_params"]
    np_ = res["normal_params"]
    numbers = (f"MILESTONE ({args.label}, iter{args.iter}, {args.task}): "
               f"TEST touch — "
               f"readable {trm*100:.2f}% ± {trs*100:.2f}% ({rp} params) vs "
               f"same-size normal {tnm*100:.2f}% ± {tns*100:.2f}% "
               f"({np_} params) -> delta {(trm-tnm)*100:+.2f} pts")
    log_test_use(time.strftime("%Y-%m-%d %H:%M"), numbers)
    print(numbers)

    gap_ok = abs(trm - tnm) <= 0.02
    beat = trm >= tnm
    gate = gap_ok and beat
    print(f"within 2 pts: {gap_ok} | readable >= normal: {beat} "
          f"| GATE: {'PASS ✔' if gate else 'not yet'}")
    if gate:
        with open(os.path.join(HERE, "STOPPED.md"), "w") as f:
            f.write("# STOPPED — rung gate passed\n\n"
                    f"{time.strftime('%Y-%m-%d %H:%M')}\n\n{numbers}\n\n"
                    "Per the mission's stop conditions: a rung's gate passes "
                    "-> stop and wait for Yahya before the next rung.\n")
        print("STOPPED.md written — waiting for Yahya before the next rung.")
        return 42
    return 0


if __name__ == "__main__":
    sys.exit(main())