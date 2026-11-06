#!/usr/bin/env python3
"""cross_eval.py — measure a model on ANOTHER corpus, by translating that corpus into the model's own
vocabulary.

WHY THIS EXISTS
Every number we have for the current corpora (~13-14% masked) lives in a different vocabulary from the
recorded numbers (54.89% small, 63.85% big), so the two cannot be compared directly -- token id 4 is a
different word in each. That has blocked the obvious question: IS our current model actually worse, or
is it being measured on something harder?

This tool removes the blocker without training anything. It takes a corpus's held-out text, translates
every token through word strings into the MODEL'S vocabulary (unknown words become <unk>, and that rate
is reported), and measures the model on it with the same assistant-masked metric. So:
  * our merged-trained model can be scored on the splitfix corpus's held-out text, and
  * the recorded splitfix model can be scored on the merged corpus's held-out text.
Both in ONE vocabulary, so the comparison is finally legal.

WHAT IS REPORTED, always together: masked accuracy, masked perplexity, the number of masked targets, and
the unknown-token rate introduced by the translation (a high rate weakens the comparison and is stated,
not hidden).

Usage: bash lens/scripts/run_cross_eval.sh <ship.npz> <vocab-corpus-dir> <val-corpus-dir> [label]
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet")
import chat_data  # noqa: E402
import lens  # noqa: E402

WIN = 32


def corpus(d):
    ids = np.load(f"{d}/ids_v12000.npy")
    b = np.load(f"{d}/bounds_v12000.npy")
    sp = np.load(f"{d}/split_v12000.npy")
    vocab = json.load(open(f"{d}/vocab_v12000.json"))
    return ids, np.asarray(b), np.asarray(sp), vocab


def main():
    ship, ship_corpus, val_corpus = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(ship)
    ship_train_corpus = sys.argv[5] if len(sys.argv) > 5 else None
    sids, sb, ssp, svocab = corpus(ship_corpus)      # gives the model's vocabulary
    vids, vb, vsp, vvocab = corpus(val_corpus)       # supplies the held-out text
    vmap = {w: i for i, w in enumerate(svocab)}
    unk = vmap.get("<unk>", 0)

    # translate the val corpus's tokens into the model's vocabulary via their word strings
    translated = np.array([vmap.get(vvocab[t], unk) for t in vids], dtype=np.int64)
    def is_assistant(tokens_words):
        """One flag PER TOKEN, including the 'human :' / 'assistant :' markers themselves. The first
        version dropped the two marker tokens from the list, so the flag list was shorter than the text
        and the target lookup ran off the end -- it crashed rather than mis-scoring, which is the good
        failure mode, but it also means the alignment has to be explicit: len(flags) == len(words)."""
        out, in_asst, k = [], False, 0
        while k < len(tokens_words):
            if (tokens_words[k] == "assistant" and k + 1 < len(tokens_words)
                    and tokens_words[k + 1] == ":"):
                in_asst = True
                out.extend([False, False])
                k += 2
                continue
            if tokens_words[k] == "human" and k + 1 < len(tokens_words) and tokens_words[k + 1] == ":":
                in_asst = False
                out.extend([False, False])
                k += 2
                continue
            out.append(in_asst)
            k += 1
        assert len(out) == len(tokens_words), f"flag misalignment {len(out)} vs {len(tokens_words)}"
        return out

    X, Y = chat_data.make_windows(translated, vb, vsp, WIN, which=1)
    # Mask computed with the PROJECT'S OWN walk, verbatim, on the translated ids -- never a second
    # hand-rolled version. Measured 2026-10-17: my earlier loop produced different window/target counts
    # than the project meter (3,724 vs 2,227 targets on the same corpus), which is exactly the failure
    # the "one implementation or the number is not the project's number" rule exists to prevent.
    is_asst = np.zeros(len(translated), dtype=np.float32)
    for j in range(len(vb) - 1):
        a0, b0 = vb[j], vb[j + 1]
        toks = [vvocab[i] for i in vids[a0:b0]]
        in_a, k = False, 0
        while k < len(toks):
            if toks[k].lower() == "assistant" and k + 1 < len(toks) and toks[k + 1] == ":":
                in_a, k = True, k + 2
                continue
            if toks[k].lower() == "human" and k + 1 < len(toks) and toks[k + 1] == ":":
                in_a, k = False, k + 2
                continue
            if in_a:
                is_asst[a0 + k] = 1.0
            k += 1
    W = is_asst[Y] > 0.5
    unk_rate = float((translated == unk).mean())

    # CROSS-CORPUS CONTAMINATION FILTER (2026-10-17). Two corpora built on the same source share text:
    # measured, the merged corpus's training text contains much of the splitfix corpus's "held-out"
    # text, which inflated a cross-corpus comparison to 55% that is NOT a generalisation number. When
    # the ship's own training corpus is given, every evaluation window that appears verbatim in that
    # corpus's TRAINING text -- translated into the same vocabulary -- is removed before scoring.
    dropped_cross = 0
    if ship_train_corpus:
        tids, tb, tsp, tvocab = corpus(ship_train_corpus)
        tmap = {w: i for i, w in enumerate(svocab)}
        translated_train = np.array([tmap.get(tvocab[t], unk) for t in tids], dtype=np.int64)
        Xtr_c, _ = chat_data.make_windows(translated_train, tb, tsp, WIN, which=0)
        def _rows(X):
            return X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
        keep = ~np.isin(_rows(X), _rows(Xtr_c))
        dropped_cross = int((~keep).sum())
        print(f"    cross-corpus filter: removed {dropped_cross:,} of {len(X):,} windows "
              f"({dropped_cross/max(len(X),1)*100:.2f}%) that appear verbatim in the ship's TRAINING text")
        X, Y, W = X[keep], Y[keep], W[keep]

    net = lens.load(ship)
    acc, corr, n, ce_sum = 0, 0, 0, 0.0
    for s in range(0, len(X), 512):
        idx = np.arange(s, min(s + 512, len(X)))
        lg = np.asarray(net.forward_arrays(X[idx])["logits"], dtype=np.float64)
        w = W[idx]
        if not w.any():
            continue
        mx = lg.max(axis=1, keepdims=True)
        logp = lg - mx - np.log(np.exp(lg - mx).sum(axis=1, keepdims=True))
        ce = -logp[np.arange(len(idx)), Y[idx]]
        ce_sum += float(ce[w].sum())
        corr += int((lg.argmax(axis=1)[w] == Y[idx][w]).sum())
        n += int(w.sum())
    masked_acc = corr / max(n, 1)
    masked_ppl = float(np.exp(ce_sum / max(n, 1)))
    print(f"  SHIP {label}   ({ship})")
    print(f"    trained on   {ship_corpus}")
    print(f"    measured on  {val_corpus} (held-out text, translated into the model's vocabulary)")
    print(f"    masked acc   {masked_acc*100:.2f}%   masked ppl {masked_ppl:.2f}   "
          f"masked targets {n:,}")
    print(f"    unk rate introduced by the translation {unk_rate*100:.2f}%  "
          f"(high would weaken the comparison; this is what it costs)")
    dest = ("/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/cross_eval_"
            f"{label}.json")
    json.dump({"ship": ship, "ship_corpus": ship_corpus, "val_corpus": val_corpus,
               "masked_acc": masked_acc, "masked_ppl": masked_ppl, "masked_n": n,
               "unk_rate": unk_rate}, open(dest, "w"), indent=1)
    print(f"    written: {dest}")


if __name__ == "__main__":
    main()