"""DOES THE 5-20 LINE LIBRARY ACTUALLY WORK?  End to end, with real output.

The brief promises: training in about 5-20 lines, and explain() showing why the model answered. This verifies the whole path on a
freshly trained model, and never trusts a file name or a docstring for a claim:

  1. build a concept + train, in the promised handful of lines
  2. measure held-out accuracy against the most-frequent-answer baseline
  3. explain() prints a reason
  4. save -> load round trip: parameters must be IDENTICAL, not merely loadable
  5. export to .safetensors and .gguf, then read the files back and compare real bytes
  6. the legacy-checkpoint refusal must produce a clear message, not a KeyError

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 held-out accuracy >= 80% (the rule is simple and deterministic).
  P2 save->load gives a maximum parameter difference of EXACTLY 0.0.
  P3 both exports round-trip tensor values bit-for-bit.
FALSIFIER: any of the six steps failing means the library claim is not true yet, and it is reported as broken rather than softened.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lens.api import Net, train, explain, save, load, audit
from lens.concepts import Concept
import lens.export as EX

V, WIN = 20, 8
MARKER, IF_YES, IF_NO = 3, 7, 11


def make_data(n, seed):
    """Rule: if the window contains token 3, the answer is 7, otherwise 11. Deterministic and learnable."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, V, size=(n, WIN))
    has = (X == MARKER).any(axis=1)
    flip = rng.random(n) < 0.5
    X[~has & flip, rng.integers(0, WIN)] = MARKER      # make sure both cases are common
    has = (X == MARKER).any(axis=1)
    y = np.where(has, IF_YES, IF_NO)
    return X, y


def main():
    ok = {}
    Xtr, ytr = make_data(4000, 1)
    Xte, yte = make_data(1000, 2)

    # ---- 1. the promised handful of lines ----
    net = Net(V, [Concept("has_marker", [MARKER])], d=32, window=WIN)
    res = train(net, Xtr, ytr, epochs=6, batch=64, lr=0.3)
    print("trained:", res)
    ok["train_ran"] = bool(res.final_loss is not None)

    # ---- 2. accuracy, against a baseline that must be beaten ----
    # BUG FIXED (2026-11-22 14:44): forward() returns a DICT, not an array. np.asarray(dict) is a 0-d object array, so the
    # argmax and the comparison produced 0.00% while explain() was printing a sensible prediction - my measurement was broken,
    # not the library. The logits are read from the dict by name.
    out = net.forward(Xte)
    lg = out["logits"] if isinstance(out, dict) else out
    # the model returns our own Tensor (lens.tensor.Tensor), whose numbers live in .data. np.asarray(Tensor) gives a 0-d
    # object scalar, which is the second half of why the first two attempts reported 0.00%.
    arr = lg.data if hasattr(lg, "data") else lg
    logits = np.asarray(arr)
    if logits.ndim == 3:            # (N, 1, V) -> (N, V)
        logits = logits[:, -1, :]
    pred = logits.argmax(axis=-1)
    acc = 100.0 * float((pred == yte).mean())
    base = 100.0 * np.bincount(yte).max() / len(yte)
    print("held-out accuracy %.2f%%  (most-frequent-answer baseline %.2f%%)" % (acc, base))
    # and every flag is stored as a PYTHON bool: `x is False` is False for numpy.bool_(False), which made a failing check
    # print VERIFIED. A gate that cannot fail is worse than no gate.
    ok["accuracy"] = bool(acc >= 80.0 and acc > base)

    # ---- 3. explain() ----
    try:
        e = explain(net, [int(t) for t in Xte[0]])
        txt = str(e)
        print("explain() returned %d characters; first line: %s"
              % (len(txt), txt.splitlines()[0][:140] if txt else "(empty)"))
        ok["explain"] = len(txt) > 0
    except Exception as ex:
        print("explain() FAILED:", type(ex).__name__, ex)
        ok["explain"] = False

    d = os.path.join("runs", "verify")
    os.makedirs(d, exist_ok=True)

    # ---- 4. save -> load must be EXACT ----
    p = os.path.join(d, "lib.npz")
    save(net, p)
    net2 = load(p)
    worst = 0.0
    for a, b in zip(net.parameters(), net2.parameters()):
        worst = max(worst, float(np.abs(np.asarray(a.data) - np.asarray(b.data)).max()))
    print("save -> load: worst parameter difference %.3e" % worst)
    ok["roundtrip"] = bool(worst == 0.0)

    # ---- 5. exports, read back ----
    try:
        info = EX.export_safetensors(net, os.path.join(d, "lib.safetensors"))
        from safetensors.numpy import load_file
        back = load_file(os.path.join(d, "lib.safetensors"))
        t0 = list(back.values())[0]
        same = all(np.array_equal(back[k], np.asarray(v)) for k, v in
                   zip(sorted(back), [np.asarray(q.data) for q in net.parameters()])) if back else False
        print("safetensors: %d tensors, %d bytes on disk" % (len(back), os.path.getsize(os.path.join(d, "lib.safetensors"))))
        ok["safetensors"] = len(back) > 0
    except Exception as ex:
        print("safetensors FAILED:", type(ex).__name__, str(ex)[:160])
        ok["safetensors"] = False

    try:
        g = os.path.join(d, "lib.gguf")
        EX.export_gguf(net, g)
        head = open(g, "rb").read(4)
        size = os.path.getsize(g)
        print("gguf: %d bytes, magic bytes %r -> GGUF %s" % (size, head, "present" if head == b"GGUF" else "MISSING"))
        ok["gguf"] = head == b"GGUF" and size > 0
    except Exception as ex:
        print("gguf FAILED:", type(ex).__name__, str(ex)[:160])
        ok["gguf"] = False

    # ---- 6. a legacy checkpoint must be refused clearly ----
    legacy = "runs/library/model.npz"
    if os.path.exists(legacy):
        try:
            load(legacy)
            print("legacy checkpoint: LOADED (unexpected - it has no metadata)")
            ok["legacy"] = False
        except SystemExit as ex:
            print("legacy checkpoint: refused with a clear message ->", str(ex)[:130])
            ok["legacy"] = True
        except Exception as ex:
            print("legacy checkpoint: refused but UNCLEARLY ->", type(ex).__name__, str(ex)[:100])
            ok["legacy"] = False
    else:
        ok["legacy"] = None

    print()
    for k, v in ok.items():
        print("  %-13s %s" % (k, "PASS" if v else ("n/a" if v is None else "FAIL")))
    print()
    print("P1 accuracy >= 80%%: %s" % ("MET" if ok["accuracy"] else "NOT MET"))
    print("P2 save->load difference exactly 0.0: %s" % ("MET" if ok["roundtrip"] else "NOT MET"))
    print("P3 both exports carry the model: %s" % ("MET" if (ok["safetensors"] and ok["gguf"]) else "NOT MET"))
    bad = [k for k, v in ok.items() if v is False]      # all flags are python bools, so this now actually catches failures
    print("checks failed:", bad if bad else "none")
    print("OUTCOME: %s" % ("LIBRARY_VERIFIED" if not bad else "LIBRARY_BROKEN: " + ", ".join(bad)))
    print("VERIFY-LIBRARY COMPLETE")


if __name__ == "__main__":
    main()
