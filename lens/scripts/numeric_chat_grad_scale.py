#!/usr/bin/env python3
"""numeric_chat_grad_scale.py - the numeric path's step size, from a MEASUREMENT not from taste.

WHY THIS EXISTS. The first two smoke runs of numeric_chat_test.py DIVERGED (nan) in the arms with the
numeric term ON, while the arm with the term OFF stayed finite at the same lr. The loss at step 0 was
total 10.05 = CE 9.65 + numeric 0.395, so the term was NOT too large in the loss - the gradient was.
mag.Wm's gradient is a sum over N*W = 8,192 rows per batch (every position of every window contributes to
the same D-vector), while the CE tensors' gradients are sums over N = 256 rows, so one lr cannot serve
both: at lr 0.2 the first step on mag.Wm was ~4 in a tensor whose initial entries are ~0.1, and the run
was nan by step 10. This prints ||g|| per tensor on ONE REAL BATCH through the REAL loss, exactly as
train_chat.py --probe-grads does for the optimiser comparison, so the numeric step can be matched to the
CE step instead of guessed. NO training happens here and NO result is produced.

Output: lens/runs/NUMERIC-CHAT-GRAD-SCALE.txt
"""
import datetime
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
os.chdir(PROJ)
os.environ["CHAT_DATA_DIR"] = os.environ.get("CHAT_DATA_DIR") or os.path.join(
    PROJ, "data", "chat", "prepared_mix_ctrl_honest")

import chat_data
from lens import backend as B
from lens.concepts import Concept
from lens.model import LensNet
from lens.numeric import magnitude_channel, numeric_targets

W, D, SCALE, BATCH = 32, 128, 100.0, 256
OUT = os.path.join(PROJ, "lens", "runs", "NUMERIC-CHAT-GRAD-SCALE.txt")
lines = []


def say(m=""):
    print(m, flush=True)
    lines.append(m)


d = chat_data.build()
ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
Xtr, Ytr = chat_data.make_windows(ids, bounds, split, W, which=0, max_windows=4000, seed=0)
id_to_token = {i: t for i, t in enumerate(vocab)}
mag, numeric = magnitude_channel(Xtr, id_to_token, scale=SCALE)
keep = 1.0 - numeric
raw, ymask = numeric_targets(vocab, Ytr, scale=1.0)
tmean = float(raw[ymask > 0.5].mean())
tstd = float(raw[ymask > 0.5].std())
ynum = ((raw - tmean) / tstd).astype(np.float32)

net = LensNet(vocab_size=len(vocab),
              concepts=[Concept("is_a_number",
                                [i for i, w in enumerate(vocab) if w.replace('.', '').isdigit()])],
              d=D, window=W, seed=0, lam_concept=0.0, lam_act=0.0, lam_w=0.0, numeric=True)
idx = np.arange(BATCH)
for p in net.parameters():
    p.grad = None
total, parts = net.loss(Xtr[idx], Ytr[idx], use_concept_loss=False,
                        mag=mag[idx], magmask=keep[idx], ynum=ynum[idx], ynum_mask=ymask[idx],
                        lam_num=1.0)
total.backward()
say("=" * 100)
say("NUMERIC PATH ON CHAT - GRADIENT SCALE PER TENSOR (one real batch, NO training, NO result)")
say(f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   corpus {os.environ['CHAT_DATA_DIR']}")
say("=" * 100)
say(f"batch {BATCH} windows of {W} tokens   loss {float(B.to_cpu(total.data)):.5f} = "
    f"CE {parts['ce']:.5f} + numeric {parts.get('num_mse', float('nan')):.5f} "
    f"(targets standardised: mean {tmean:.3f} std {tstd:.3f})")
say("")
say(f"{'tensor':<16} {'n':>10} {'||g||':>12} {'rms entry':>12}   role")
g = {}
for p in net.parameters():
    if p.grad is None:
        continue
    gg = np.asarray(B.to_cpu(p.grad), dtype=np.float64)
    g[p.name] = float(np.linalg.norm(gg))
    role = "numeric path" if p.name.startswith(("mag.", "num.")) else "language path"
    say(f"{p.name:<16} {gg.size:>10,} {np.linalg.norm(gg):>12.4e} "
        f"{np.sqrt((gg**2).mean()):>12.4e}   {role}")
lang = [v for k, v in g.items() if not k.startswith(("mag.", "num."))]
say("")
say(f"median ||g|| over the LANGUAGE tensors: {np.median(lang):.6g}")
say("")
say("THE RULE THIS MEASUREMENT FIXES (declared in NUMERIC-CHAT-PREDICTIONS.txt ADDENDUM 3, before the run):")
say("  each numeric-path tensor gets its OWN step size, chosen so that its FIRST step moves it as far as")
say("  the language tensors' first step at the run's lr (0.2):   lr_num = 0.2 * median(||g_lang||) / ||g_num||")
for k in ("mag.Wm", "mag.bm", "num.Wn", "num.bn"):
    if k in g:
        say(f"    {k:<10} ||g|| {g[k]:.6g}  ->  lr {0.2 * np.median(lang) / g[k]:.6g}")
say("")
say("A tensor with no gradient is not printed (it would take no step anyway). This file trains nothing.")

with open(OUT, "a") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nappended to {OUT}")
