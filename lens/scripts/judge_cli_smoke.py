"""judge_cli_smoke.py — prove the shipped judge calls corruptions BAD and real data GOOD,
printing the arithmetic for each. In-process (no shell), so nothing can hide in a pipe.
"""
import os
import sys

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, P)
sys.path.insert(0, os.path.join(P, "lens", "scripts"))

import chat_data                                                    # noqa: E402
from judge_cli import corpus_context, load_judge, score_windows, text_to_window  # noqa: E402


def show(tag, X, ctx, W, b, feats):
    F, p = score_windows(X, ctx, W, b, feats)
    verdict = "GOOD" if p[0] >= 0.5 else "BAD"
    top = sorted(zip(feats, F[0], F[0] * W), key=lambda t: -abs(t[2]))[:3]
    print(f"  {tag:<22} score {p[0]:.4f} -> {verdict:<4} "
          f"top: " + "  ".join(f"{n}={v:.2f}({c:+.2f})" for n, v, c in top))
    return verdict


def main():
    W, b, feats = load_judge()
    ctx = corpus_context()
    vocab = ctx[0]
    d = chat_data.build()
    Xva, _ = chat_data.make_windows(d["ids"], d["bounds"], d["split"], 32, which=1)
    real = np.asarray(Xva[7], dtype=np.int64).reshape(1, -1)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(real[0]).reshape(1, -1)
    blank = np.zeros((1, 32), dtype=np.int64)
    text = text_to_window("hello how are you today i am fine thank you", vocab)

    print("judge_cli smoke — real vs corruptions vs typed text")
    v1 = show("real val window", real, ctx, W, b, feats)
    v2 = show("same window SHUFFLED", shuffled, ctx, W, b, feats)
    v3 = show("BLANK window", blank, ctx, W, b, feats)
    v4 = show("typed sentence", text, ctx, W, b, feats)
    ok = (v1 == "GOOD") and (v2 == "BAD") and (v3 == "BAD")
    print(f"\n  smoke: {'PASS' if ok else 'FAIL'} (real GOOD, both corruptions BAD; "
          f"typed sentence -> {v4})")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())