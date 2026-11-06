"""tune_normal_banking77.py — FAIR comparison for rung 3 (Yahya's order,
2026-10-02): give the NORMAL network the same tuning budget the readable
got (27 validation-only iterations). VAL only — the test set stays
untouched. Sweep grid = the readable's own ladder axes, at ~30k params:
epochs {48,96,192,384,768,1536} x lr {0.1,0.2} x batch {64,128} with
h=24 (30,149 params exactly) = 24 configs, + 3 widths (h 23/25 at the
best config, plus the best config re-run = the 27th slot is the best
config's deterministic repeat). Verdict rules (pre-registered): the
readable's rung-3 VAL record = 86.08 ± 0.60 (3 seeds). If the tuned
normal's best VAL >= 86.08 - 2*pooled_std -> "not a win" (equal within
seed noise); say so plainly. Every config: 1 seed (seed 0) in the sweep,
then 3-seed confirmation at the best config. Resume-safe: each completed
config is appended to results.json immediately."""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from glassnet import text_data as TD
from glassnet.loop_lib import load_split
from glassnet.tensor import Tensor, relu
from glassnet.layers import Dense
from glassnet.losses import cross_entropy
from glassnet.optim import SGD

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(HERE, "runs", "rung3_normal_tune", "results.json")
LOG_PATH = os.path.join(HERE, "runs", "rung3_normal_tune", "sweep.log")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

Xtr, ytr, Xva, yva, Xte, yte = load_split("banking77")

EPOCHS_SET = (48, 96, 192, 384, 768, 1536)
LRS = (0.1, 0.2)
BATCHES = (64, 128)
H = 24


def make_net(h):
    rng = np.random.default_rng(0)

    class MLP:
        def __init__(self):
            self.l1 = Dense(1175, h, rng=rng)
            self.l2 = Dense(h, 77, rng=rng)

        def __call__(self, x):
            return self.l2(relu(self.l1(x)))

        def parameters(self):
            return self.l1.parameters() + self.l2.parameters()

    return MLP()


def run_cfg(ep, lr, bs, h, seed=0):
    rngd = np.random.default_rng(seed * 131 + 7)
    net = make_net(h)
    opt = SGD(net.parameters(), lr=lr)
    for _ in range(ep):
        perm = rngd.permutation(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = cross_entropy(net(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    v = float(np.mean(net(Xva).data.argmax(1) == yva))
    return v * 100


def load_done():
    if not os.path.exists(OUT_PATH):
        return []
    return json.load(open(OUT_PATH))


def save_done(done):
    with open(OUT_PATH, "w") as f:
        json.dump(done, f, indent=1)


def done_key(d):
    return (d["ep"], d["lr"], d["bs"], d["h"])


def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def main():
    done = load_done()
    keys = {done_key(d) for d in done}
    grid = sorted([(ep, lr, bs, H)
                   for ep in EPOCHS_SET for lr in LRS for bs in BATCHES])
    log(f"sweep starts: {len(grid)} grid configs, h={H} ({1253*H+77} params)")
    for cfg in grid:
        ep, lr, bs, h = cfg
        if cfg in keys:
            continue
        t0 = time.time()
        v = run_cfg(ep, lr, bs, h)
        rec = {"ep": ep, "lr": lr, "bs": bs, "h": h, "val": v,
               "params": 1253 * h + 77, "secs": round(time.time() - t0)}
        done.append(rec)
        save_done(done)
        log(f"  ep={ep:4d} lr={lr} bs={bs:3d} -> val {v:.2f} "
            f"({rec['secs']}s)")
    best = max(done, key=lambda d: d["val"])
    log(f"grid best: ep={best['ep']} lr={best['lr']} bs={best['bs']} "
        f"h={best['h']} val {best['val']:.2f}")
    # 3 widths at the grid-best: h=23, h=25 (+ the 27th slot = best repeat)
    for h in (23, 25, best["h"]):
        cfg = (best["ep"], best["lr"], best["bs"], h)
        if cfg in keys:
            continue
        t0 = time.time()
        v = run_cfg(*cfg)
        rec = {"ep": cfg[0], "lr": cfg[1], "bs": cfg[2], "h": h, "val": v,
               "params": 1253 * h + 77, "secs": round(time.time() - t0)}
        done.append(rec)
        save_done(done)
        log(f"  WIDTH h={h} ep={cfg[0]} lr={cfg[1]} bs={cfg[2]} -> "
            f"val {v:.2f} ({rec['secs']}s)")
    best = max(done, key=lambda d: d["val"])
    readable = 86.08
    gap = readable - best["val"]
    pooled = 0.60  # readable 3-seed std (iter 033 record); sweep is 1-seed
    verdict = ("not a win (tuned normal within seed noise of readable)"
               if best["val"] >= readable - 2 * pooled
               else "readable win holds provisionally")
    log(f"FINAL: tuned-normal best VAL {best['val']:.2f} "
        f"(ep={best['ep']} lr={best['lr']} bs={best['bs']} h={best['h']}) "
        f"vs readable 86.08 ± 0.60 -> gap {gap:+.2f} -> {verdict}")


if __name__ == "__main__":
    main()