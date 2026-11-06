#!/bin/bash
# HONEST EVAL — any talk ship, measured on the CURATED (deduplicated) val split.
# Usage: bash lens/scripts/run_eval_clean.sh <ship-relative-path>
# Why: 22.2% of the original val split shared a 100-token window with train, so every number
# measured on it (30.94% plain, 80.59% assistant-masked) is inflated by memorised templates.
# Metrics (identical definitions to the record): assistant-masked acc + ppl, and plain-pool acc,
# over the FULL clean val split (not the first 4000 windows).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
# EVAL_DATA_DIR lets the SAME metric run on another corpus (e.g. the merged external corpus).
# Default is unchanged, so every earlier number reproduces exactly.
export CHAT_DATA_DIR="${EVAL_DATA_DIR:-$PWD/data/chat/prepared_clean}"
SHIP="${1:?usage: run_eval_clean.sh <ship>}"
.venv-gpu/bin/python3 -u - "$SHIP" <<'PY'
import os, sys, json
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")
import numpy as np
import chat_data, lens
ship = sys.argv[1]
WIN = 32
d = chat_data.build()
ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
Xva, Yva = chat_data.make_windows(ids, bounds, split, WIN, which=1)
# OPTIONAL CONTAMINATION FILTER (2026-10-17). A val window that appears VERBATIM in the same corpus's
# training text is not held out at all: the model has seen those exact 32 tokens. Measured on the
# recorded 63.85%: 63.21% of the val windows that produced it were present verbatim in the training text
# of the corpus the model actually trained on. Set EVAL_DROP_TRAIN_OVERLAP=1 to remove them and get the
# number that is really held out. Default is OFF, so every earlier number still reproduces exactly.
if os.environ.get("EVAL_DROP_TRAIN_OVERLAP") == "1":
    Xtr_all, _ = chat_data.make_windows(ids, bounds, split, WIN, which=0)
    def _rows(X):
        return X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
    keep = ~np.isin(_rows(Xva), _rows(Xtr_all))
    print(f"  contamination filter: dropping {int((~keep).sum()):,} of {len(Xva):,} val windows "
          f"({(~keep).mean()*100:.2f}%) that appear verbatim in this corpus's TRAIN text")
    Xva, Yva = Xva[keep], Yva[keep]
# assistant mask on val targets (same walk as the trainer/distill script)
is_asst = np.zeros(len(ids), dtype=np.float32)
for j in range(len(bounds) - 1):
    a0, b0 = bounds[j], bounds[j + 1]
    toks = [vocab[i] for i in ids[a0:b0]]
    in_a, k = False, 0
    while k < len(toks):
        role = toks[k].lower()
        # TWO TURN FORMATS. The old corpora write "assistant :", but the ChatML corpora write
        # "<|im_start|> assistant" with NO colon. Only the first form was recognised, so on the ChatML
        # corpus every window counted as non-assistant, the masked pass skipped every batch, and the eval
        # died with "need at least one array to concatenate" instead of printing an accuracy.
        prev_ctrl = k > 0 and toks[k - 1] == "<|im_start|>"
        colon = k + 1 < len(toks) and toks[k + 1] == ":"
        if role in ("assistant", "human", "user") and (prev_ctrl or colon):
            in_a = (role == "assistant")
            k += 2 if colon else 1
            continue
        if in_a:
            is_asst[a0 + k] = 1.0
        k += 1
Wva = is_asst[Yva]
net = lens.load(os.path.join("/run/media/yahya/HARD DISK 4/GlassNet", ship), device="gpu")
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
def evaluate(mask):
    P, C, S = [], 0, 0
    for s in range(0, len(Xva), 512):
        idx = np.arange(s, min(s + 512, len(Xva)))
        p = softmax(net.forward_arrays(Xva[idx])["logits"])
        t = Yva[idx]
        if mask:
            m = Wva[idx] > 0.5
            if m.sum() == 0:
                continue
            p, t = p[m], t[m]
        P.append(-np.log(np.maximum(p[np.arange(len(t)), t], 1e-12)))
        C += int((p.argmax(axis=1) == t).sum())
        S += len(t)
    if not P:
        # nothing matched this mask (e.g. a corpus whose turn format is not recognised). Say so with the
        # count, instead of dying on an empty concatenate and losing the whole measurement.
        print(f"  WARNING: no windows matched mask={mask} (0 of {len(Xva)}): reporting nan, not a crash")
        return float("nan"), float("nan"), 0
    P = np.concatenate(P)
    return float(np.exp(P.mean())), C / max(S, 1), S
mp, ma, ms = evaluate(True)
pp, pa, ps = evaluate(False)
RAW_N, Wva_raw = len(Xva), Wva
print(f"  SHIP {ship.split('/')[-1]}")
print(f"  RAW (as recorded; includes any memorised val windows) n={RAW_N} windows "
      f"(assistant targets {int((Wva_raw>0.5).sum())}):")
print(f"    assistant-masked: acc {ma*100:.2f}%  ppl {mp:.3f}   (n={ms})")
print(f"    plain pool:       acc {pa*100:.2f}%  ppl {pp:.1f}   (n={ps})")

# HONEST NUMBER, always computed and always printed beside the raw one (2026-10-17). A val window that
# appears verbatim in this corpus's training text is not held out; the gap between these two lines is
# exactly how much of the number came from memorisation. Printing both is deliberate: a run must never
# be able to report only the flattering line.
# THE LEAK CHECK IS EXPENSIVE AND OPTIONAL. Uncapped make_windows enumerates all 16,878,132 train windows
# (4.3 GB) and np.isin sorts them, which is how this eval got OOM-killed (exit 137). A corpus that has
# already passed an exact-byte fixed-point repair (VERDICT CLEAN, 0 overlapping windows -- see
# lens/runs/REPAIR-CTRL.txt and CLEAN-VAL-MIX.txt) does not need it repeated here: set SKIP_LEAK_CHECK=1.
# It is NOT skipped silently -- the run prints which mode it used.
if os.environ.get("SKIP_LEAK_CHECK") == "1":
    print("  LEAK CHECK SKIPPED on request (SKIP_LEAK_CHECK=1): this corpus passed an exact-byte fixed-point")
    print("  repair with 0 overlapping windows, so the raw line above IS the held-out number.")
    Xtr_all2 = np.zeros((0, WIN), dtype=np.int64)
else:
    Xtr_all2, _ = chat_data.make_windows(ids, bounds, split, WIN, which=0)
def _rows2(X):
    return X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
RAW_N = len(Xva)
if len(Xtr_all2):
    keep2 = ~np.isin(_rows2(Xva), _rows2(Xtr_all2))
    dropped = int((~keep2).sum())
    if dropped:
        Xva_h, Yva_h = Xva[keep2], Yva[keep2]
        Xva, Yva = Xva_h, Yva_h
        mp2, ma2, ms2 = evaluate(True)
        pp2, pa2, ps2 = evaluate(False)
        print(f"  HONEST (val windows present verbatim in this corpus's TRAIN text removed: "
              f"{dropped:,} of {RAW_N:,} = {dropped/RAW_N*100:.2f}%):")
        print(f"    assistant-masked: acc {ma2*100:.2f}%  ppl {mp2:.3f}   (n={ms2})")
        print(f"    plain pool:       acc {pa2*100:.2f}%  ppl {pp2:.1f}   (n={ps2})")
    else:
        print("  HONEST: no val window appears in this corpus's train text (0 dropped) -- "
              "the raw line above IS the held-out number")
PY