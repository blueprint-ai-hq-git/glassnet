#!/usr/bin/env python3
"""w64_structural_check.py — CAN THE TIED MODEL BE BUILT AND TRAINED AT WINDOW 64?
NO CORPUS IS LOADED, NO GPU, NO TRAINING RUN. This is a WIRING/STRUCTURE check only: it answers
whether --window 64 needs any code change (it should not: every window-dependent tensor is built from
W) and what the exact parameter counts are, so P95a's promised numbers can be falsified in the log.

Run:  .venv-gpu/bin/python3 lens/scripts/w64_structural_check.py   (or system python3; the engine falls
      back to numpy when no CUDA device is used)
Writes: lens/runs/W64-STRUCTURAL.txt
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
os.chdir(PROJ)
sys.path.insert(0, PROJ)

import lens.api as api                                    # noqa: E402
from lens.concepts import Concept                          # noqa: E402
import lens.tensor as T                                    # noqa: E402

VOCAB = 12002          # the chat corpus vocabulary
K = 36                 # the concepts the full-corpus arms keep after the 0.005 support floor
D = 256
OUT = os.path.join(PROJ, "lens", "runs", "W64-STRUCTURAL.txt")

lines = []


def say(s=""):
    print(s)
    lines.append(s)


def concepts(n=K):
    """Synthetic concepts with the real K: the parameter count depends on K, V, D and W only.
    Supports are nominal (they only scale pos_weight); NO corpus is touched."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        ids = sorted(set(int(x) for x in rng.integers(1, VOCAB, size=6)))
        out.append(Concept(f"c{i:02d}", ids, support=0.05))
    return out


say("=" * 94)
say("W64 STRUCTURAL CHECK — no corpus, no GPU, no training. Only: does W=64 build, forward, backward?")
say("=" * 94)
say(f"vocab {VOCAB}  K {K}  d {D}  windows compared: 32 (the twin) and 64 (the candidate)")
say("")

counts = {}
grad_ok = {}
for W in (32, 64):
    net = api.Net(VOCAB, concepts(), d=D, window=W, seed=0, device="cpu",
                  direct=True, direct_mode="tied")
    npar = sum(int(np.prod(p.data.shape)) for p in net.parameters())
    counts[W] = npar
    say(f"  window {W:>2}: model built, {npar:,} params")

    # a forward + a real backward on synthetic ids (never corpus text)
    rng = np.random.default_rng(7)
    X = rng.integers(0, VOCAB, size=(8, W)).astype(np.int32)
    Y = rng.integers(0, VOCAB, size=8).astype(np.int32)
    out = net.forward(X)
    logits = out["logits"]
    say(f"           logits shape {tuple(np.asarray(logits.data).shape)}  "
        f"(expected (8, {VOCAB}))")
    loss = T.softmax_ce(logits, Y)
    lv = float(np.asarray(loss.data))
    loss.backward()
    dmix = net.parameters()[-1]
    dg = np.asarray(dmix.grad)
    eg = np.asarray(net.parameters()[0].grad)
    finite = bool(np.isfinite(lv) and np.isfinite(dg).all() and np.isfinite(eg).all())
    nonzero = bool(np.abs(dg).max() > 0 and np.abs(eg).max() > 0)
    grad_ok[W] = (finite, nonzero)
    say(f"           synthetic loss {lv:.4f} finite={np.isfinite(lv)}  "
        f"grad dmix max|.| {np.abs(dg).max():.3e}  grad emb max|.| {np.abs(eg).max():.3e}  "
        f"finite={finite} nonzero={nonzero}")

    # the window guard must still FIRE: a wrong-width window has to be refused, or the flag is not what
    # decides the width
    wrong = W + 1
    try:
        net.forward(rng.integers(0, VOCAB, size=(2, wrong)).astype(np.int32))
        guard = "DID NOT FIRE (!!)"
    except ValueError as e:
        guard = f"fired: {e}"
    say(f"           window guard on a {wrong}-wide input: {guard}")

say("")
delta = counts[64] - counts[32]
dmix_delta = (D * 64 * D) - (D * 32 * D)
concept_delta = (D * 64 * K) - (D * 32 * K)
say(f"PARAM DELTA window32 -> window64: {delta:,}   (= dmix +{dmix_delta:,} "
    f"+ concept.W +{concept_delta:,} = {dmix_delta + concept_delta:,})")
say(f"  predicted count at window 64 for the real model (V={VOCAB}, K={K}, d={D}): {counts[64]:,}")
say(f"  the stored window-32 twin's own log line is 5,908,686 "
    f"[lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt line 61]; here at window 32: {counts[32]:,}")
say("")
say(f"VERDICT P95a PRE-CHECK: builds at both widths {counts[32] == 5908686 or 'NOTE: differs from the twin log'} ; "
    f"dmix and emb grads finite and non-zero at window 64: {grad_ok[64] == (True, True)}")
say("")
say("WHAT THIS DOES NOT SHOW, STATED SO IT CANNOT BE MISREAD: it is a WIRING check on synthetic ids. It says")
say("nothing about quality, accuracy or talk. It does not load the corpus, so it does not test the data build")
say("(the window-64 make_windows path is exercised by the arm itself) and it does not test memory.")
open(OUT, "w").write("\n".join(lines) + "\n")
say(f"written: {OUT}")
