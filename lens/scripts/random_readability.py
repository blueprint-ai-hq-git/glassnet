#!/usr/bin/env python3
"""random_readability.py - the readability gate, run on RANDOM inputs.

Why it exists: the user's requirement is that a normal person trying this gets 100% readability, not a
hand-picked demo. So nothing here is curated. Windows are drawn uniformly at random from the held-out
split with a recorded seed, and every claim is checked on those draws.

Three checks per window, all binary:
 1. EXPLAIN IS EXACT: explain()'s reconstruction (sum of contributions + bias) must equal the model's
    own score, residual exactly 0.0. Anything else means the printed reason is not the real reason.
 2. NO FALSE FIRES: if a named unit fires (h > 0.5), the window must actually contain that unit's
    declared tokens. A unit that fires without its evidence is a lie with a name on it.
 3. THE ANSWER COMES FROM THE NAMED UNITS: zeroing the concept vector must change the top prediction.
    If it does not, the answer is being computed somewhere else and the model is not readable.

Reported but not gated: how many units fire per window, and the honest-fire rate (a unit staying silent
is allowed; a unit lying is not).

Usage: SHIP=... N=200 SEED=20261018 python3 lens/scripts/random_readability.py
"""
import json
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
sys.path.insert(0, f"{ROOT}/lens/scripts")
os.chdir(ROOT)

import lens  # noqa: E402
import chat_data  # noqa: E402
from lens.concepts import explain as explain_fn  # noqa: E402

SHIP = os.environ.get("SHIP", "lens/runs/chat/sb_S0_d256_noattn_ep96.npz")
N = int(os.environ.get("N", "200"))
SEED = int(os.environ.get("SEED", "20261018"))

net = lens.load(SHIP, device="cpu")
d = chat_data.build()
vocab = d["vocab"]
ids = np.asarray(d["ids"], dtype=np.int64)
bounds, split = d["bounds"], d["split"]

Xva, _Yva = chat_data.make_windows(ids, bounds, split, net.W, which=1)
print(f"readability gate on {os.path.basename(SHIP)}")
print(f"  held-out windows available: {len(Xva):,}   drawing {N} UNIFORMLY AT RANDOM (seed {SEED})")
rng = np.random.default_rng(SEED)
idx = rng.choice(len(Xva), size=min(N, len(Xva)), replace=False)
X = Xva[idx]

# ---- 1. explain exactness + 3. counterfactual, over the random draw
res_max, changed, fires_per_window = 0.0, 0, []
false_fires, honest_fires = [], []
bias_only = np.asarray(net.answer.b.data, dtype=np.float64)

for w in X:
    e = explain_fn(net, w, vocab=None, top_k=6)
    res_max = max(res_max, abs(float(e.get("residual", 0.0))))
    real_top = int(np.argmax(np.asarray(net.forward_arrays(w[None, :])["logits"])[0]))
    # zeroed concepts -> the answer layer can only return its bias; that is what the model would say
    # if its named units said nothing
    if int(np.argmax(bias_only)) != real_top:
        changed += 1
    h = np.asarray(net.forward_arrays(w[None, :])["concepts"])[0]
    fired = [i for i in range(len(h)) if h[i] > 0.5]
    fires_per_window.append(len(fired))
    wt = set(int(t) for t in w)
    for i in fired:
        declared = set(int(t) for t in np.asarray(net.concepts[i].ids).reshape(-1))
        if declared & wt:
            honest_fires.append(net.concepts[i].name)
        else:
            false_fires.append(net.concepts[i].name)

n = len(X)
explain_ok = res_max == 0.0
no_false = len(false_fires) == 0
answer_from_concepts = changed == n

print(f"  1. explain exact on every random window: residual max {res_max:.1e}  "
      f"[{'PASS' if explain_ok else 'FAIL'}]")
print(f"  2. false fires (a unit firing without its own evidence): {len(false_fires)}  "
      f"[{'PASS' if no_false else 'FAIL'}]")
print(f"  3. answer changes when the named units are zeroed: {changed}/{n}  "
      f"[{'PASS' if answer_from_concepts else 'FAIL'}]")
print(f"  info: units firing per window mean {np.mean(fires_per_window):.2f} "
      f"(min {min(fires_per_window)}, max {max(fires_per_window)})")
print(f"  info: honest fires {len(honest_fires)}, i.e. "
      f"{len(honest_fires)/max(len(honest_fires)+len(false_fires),1)*100:.1f}% of all fires had their "
      f"own evidence present")
if false_fires:
    from collections import Counter
    print(f"  the lying units: {Counter(false_fires).most_common(6)}")

verdict = "PASS" if (explain_ok and no_false and answer_from_concepts) else "FAIL"
print(f"\n  READABILITY GATE (random draw, seed {SEED}, n={n}): {verdict}")
out = {"ship": SHIP, "seed": SEED, "n": n, "explain_residual_max": res_max,
       "false_fires": false_fires, "honest_fires": len(honest_fires),
       "answer_changed_when_concepts_zeroed": changed,
       "units_firing_per_window_mean": float(np.mean(fires_per_window)), "verdict": verdict}
json.dump(out, open(f"lens/runs/readability_random_{os.path.basename(SHIP).replace('.npz','')}.json", "w"),
          indent=1)
sys.exit(0 if verdict == "PASS" else 1)