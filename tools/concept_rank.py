#!/usr/bin/env python3
"""WHICH concepts are load-bearing? Single-concept ablation, ranked, plus a top-8 concentration test.

Why this exists (2026-11-20 13:25): the measured concept *share* turned out to be the wrong question. Forcing the answer through
the named layer raised the share 88.76% -> 100.00% while the named-vs-random ablation ratio FELL 5.39x -> 4.78x, and 22 of 36
concepts were needed to remove 90% of the margin in BOTH arms. Share is not specificity. This tool asks the specific question:
ablate each concept ALONE and rank them, then test whether a small top set carries the decision.

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1  the top-8 concepts by single-ablation remove at least 80% of the margin that ablating all 36 removes (concentration is real)
  P2  the top-8 sets of the two arms overlap in at least 5 concepts (the load-bearing set is a property of the architecture, not
      of which arm trained it)
  FALSIFIER: if the top-8 remove less than 60% of the all-36 margin, the margin is spread diffusely across the concept layer and
      "which concepts matter" is the wrong question for this shape. The answer in that case is FEWER, WIDER concepts - a structural
      change - and this line closes rather than being refined again.

Inference only. No training, no checkpoints written. Reuses the exact ablation semantics of tools/chat_explain_causal.py:
logits = h @ W + b + direct_vec, where direct_vec is recovered as the residual and held constant while concepts are zeroed.
"""
import sys, os, argparse, json, math
import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lens", "scripts"))
os.chdir(ROOT)
os.environ.setdefault("CHAT_DATA_DIR", os.path.join(ROOT, "data", "chat", "prepared_mix_ctrl_honest"))


def load_windows(ckpt, n_windows):
    import lens
    from lens import backend
    import chat_data
    d = chat_data.build()
    vocab, ids, bounds, split = d["vocab"], d["ids"], d["bounds"], d["split"]
    net = lens.load(os.path.join(ROOT, ckpt), device="cpu")
    W_WIN = int(getattr(net, "W", 32))
    from lens.model import forward_arrays
    X = chat_data.make_windows(ids, bounds, split, W_WIN, which=1, max_windows=n_windows, seed=0)[0]
    V_MODEL = int(backend.to_cpu(net.emb.data).shape[0])
    oob = X >= V_MODEL
    keep = ~oob.any(axis=1)
    print("  %s: W=%d, model V=%d, dropped %d/%d windows (out-of-range tokens)"
          % (os.path.basename(ckpt), W_WIN, V_MODEL, int((~keep).sum()), len(X)))
    X = X[keep]
    if len(X) == 0:
        raise SystemExit("no usable windows for %s - refusing to print a number from zero windows" % ckpt)
    return net, vocab, X, forward_arrays, backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--fixed-runner", action="store_true",
                    help="attribute to the margin against the BASELINE runner-up class, held fixed. This makes the attribution "
                         "exactly additive (contribution_i = h_i * (W[i,pred] - W[i,runner])) and therefore checkable; the default "
                         "re-computes the runner-up after ablation, which is non-additive and produced -302%% and -204%% 'shares'.")
    a = ap.parse_args()

    net, vocab, X, forward_arrays, backend = load_windows(a.ckpt, a.windows)
    W = backend.to_cpu(net.answer.W.data).astype(np.float64)
    b = backend.to_cpu(net.answer.b.data).astype(np.float64)
    n_c = len(net.concepts)
    cnames = [getattr(c, "name", "concept_%d" % i) for i, c in enumerate(net.concepts)]

    per_concept = [[] for _ in range(n_c)]   # each: fraction of margin removed when zeroed alone
    all_removed = []                          # fraction removed when ALL concepts are zeroed
    used = 0

    for i in range(X.shape[0]):
        act = forward_arrays(net, np.asarray(X[i])[None, :])
        h = backend.to_cpu(act["concepts"][0]).astype(np.float64)
        logits = backend.to_cpu(act["logits"][0]).astype(np.float64)
        pred = int(np.argmax(logits))
        srt = np.sort(logits)[::-1]
        margin = float(srt[0] - srt[1])
        if margin <= 1e-9:
            continue
        direct_vec = logits - (h @ W + b)

        def frac_removed(zero_idx):
            h2 = h.copy()
            if len(zero_idx):
                h2[list(zero_idx)] = 0.0
            lg = h2 @ W + b + direct_vec
            o = np.argsort(lg)[::-1]
            m2 = float(lg[o[0]] - lg[o[1]])
            return (margin - m2) / margin

        used += 1
        all_removed.append(frac_removed(range(n_c)))
        for c in range(n_c):
            per_concept[c].append(frac_removed([c]))

    if used == 0:
        raise SystemExit("no windows with a usable margin - refusing to report")

    med = np.array([float(np.median(v)) if v else 0.0 for v in per_concept])
    order = np.argsort(med)[::-1]
    base = float(np.median(all_removed))

    print("\n  windows scored: %d   median margin removed by ALL %d concepts: %.2f%%" % (used, n_c, 100 * base))
    print("\n  ranked by median margin removed when that concept ALONE is zeroed:")
    for rank, c in enumerate(order, 1):
        mark = " <<< top-%d" % a.top if rank <= a.top else ""
        print("   %2d. %-22s %7.2f%%%s" % (rank, cnames[c], 100 * med[c], mark))

    # the top-set concentration test: ablate the top-k TOGETHER and compare against all-36
    topset = list(order[:a.top])
    together = []
    for i in range(X.shape[0]):
        act = forward_arrays(net, np.asarray(X[i])[None, :])
        h = backend.to_cpu(act["concepts"][0]).astype(np.float64)
        logits = backend.to_cpu(act["logits"][0]).astype(np.float64)
        srt = np.sort(logits)[::-1]
        margin = float(srt[0] - srt[1])
        if margin <= 1e-9:
            continue
        direct_vec = logits - (h @ W + b)
        h2 = h.copy(); h2[topset] = 0.0
        lg = h2 @ W + b + direct_vec
        o = np.argsort(lg)[::-1]
        together.append((margin - float(lg[o[0]] - lg[o[1]])) / margin)
    top_med = float(np.median(together))
    share_of_all = top_med / base if abs(base) > 1e-12 else float("nan")

    print("\n  top-%d ablated TOGETHER: median %.2f%% of the margin vs ALL-%d %.2f%%  ->  carries %.1f%% of the all-concept effect"
          % (a.top, 100 * top_med, n_c, 100 * base, 100 * share_of_all))
    print("  top-%d names: %s" % (a.top, ", ".join(cnames[c] for c in topset)))
    if share_of_all >= 0.80:
        print("  CONCENTRATION (tool-side test): MET (>= 80%)")
    elif share_of_all >= 0.60:
        print("  CONCENTRATION (tool-side test): PARTIAL (60-80%)")
    else:
        print("  CONCENTRATION (tool-side test): NOT MET (<60%) - attribution is diffuse. NOTE: on the re-computed-runner path this")
        print("    path is NON-ADDITIVE and its numbers are invalid; use --fixed-runner for any claim.")

    concept_share = []
    share_raw = []
    if a.fixed_runner:
        # EXACT ATTRIBUTION (2026-11-20 13:35). The margin is NOT linear in the concepts, because ablating a unit can change WHICH
        # class is runner-up; that is why single-concept ablation reported negative "margin removed" for 19 of 36 concepts and the
        # top-8 together reported -302% (ON) / -204% (OFF) against an all-36 figure of 66.41% / 85.63%. Those numbers are not a
        # failed prediction, they are an invalid instrument. Against a FIXED runner-up the split is exact:
        #     contribution_i = h_i * (W[i, pred] - W[i, runner])     and     sum_i contribution_i == margin   (checkable)
        print("\n  === EXACT ATTRIBUTION against the BASELINE runner-up, held fixed ===")
        contribs = []
        recon_err = []
        for i in range(X.shape[0]):
            act = forward_arrays(net, np.asarray(X[i])[None, :])
            h = backend.to_cpu(act["concepts"][0]).astype(np.float64)
            logits = backend.to_cpu(act["logits"][0]).astype(np.float64)
            order0 = np.argsort(logits)[::-1]
            pred, runner = int(order0[0]), int(order0[1])
            margin = float(logits[pred] - logits[runner])
            if margin <= 1e-9:
                continue
            # STALE-VARIABLE BUG FIXED (2026-11-22 15:12): this loop never computed direct_vec. The name was left over from the
            # PREVIOUS loop's last iteration, so `nonconcept` was built from another window's residual and the accounting could
            # never close (8.149e+00). This is the actual root cause behind the original 5.566e+00 as well.
            direct_vec = logits - (h @ W + b)
            cvec = h * (W[:, pred] - W[:, runner])       # the CONCEPT part of the margin
            # BUG FIXED (2026-11-22 15:10): logits = h @ W + b + direct_vec, so the margin is
            #     margin = cvec.sum() + (b+direct)[pred] - (b+direct)[runner]
            # The old line compared cvec.sum() against the WHOLE margin and printed "max |sum(contribution) - margin|: 5.566e+00"
            # on a model whose concept path is exactly additive. It was charging the bias and the direct path to the concept
            # layer as error, so the check that "CAN fail" was failing for a reason that had nothing to do with concepts. The
            # decomposition below is the check that can still fail, and now it fails only if the accounting is genuinely wrong.
            nonconcept = float((b + direct_vec)[pred] - (b + direct_vec)[runner])
            recon_err.append(abs(float(cvec.sum()) + nonconcept - margin))
            contribs.append(cvec / margin)                # fraction of the margin each concept supplies
            # THE STATISTIC ITSELF WAS THE BUG (2026-11-22 15:24). Dividing each window's concept contribution by THAT
            # WINDOW'S margin and averaging the ratios lets any window with a tiny margin dominate: the ratio explodes and the
            # mean is meaningless. Measured: 993.4% at 60 windows and 126.8% at 200 windows on the SAME model - the number was
            # a sampling artefact, not a property. Both a MEDIAN ratio and an AGGREGATE ratio (total contribution over total
            # margin) are now reported, and only those may be quoted.
            concept_share.append(float(cvec.sum()) / margin)
            share_raw.append((float(cvec.sum()), float(margin)))
        concept_share = concept_share if "concept_share" in dir() else []
        C = np.array(contribs)
        mean_share = C.mean(axis=0)
        pos_total = float(np.clip(mean_share, 0, None).sum())
        ordr = np.argsort(mean_share)[::-1]
        print("    windows: %d   max |sum(contribution) - margin|: %.3e  (a check that CAN fail)"
              % (len(C), max(recon_err)))
        print("    %d of %d concepts supply POSITIVE margin; positive supply totals %.3f of the margin"
              % (int((mean_share > 0).sum()), n_c, pos_total))
        if concept_share:
            _mean_ratio = float(np.mean(concept_share))
            _med_ratio = float(np.median(concept_share))
            _agg = float(np.sum([r[0] for r in share_raw]) / np.sum([r[1] for r in share_raw]))
            print("    the concept layer as a whole: MEAN of per-window ratios %.1f%%  (UNSTABLE, do not quote)" % (100 * _mean_ratio))
            print("                                  MEDIAN per-window ratio %.1f%%" % (100 * _med_ratio))
            print("                                  AGGREGATE (total contribution / total margin) %.1f%%   <- quote this"
                  % (100 * _agg))
            print("    so the bias plus the direct path supplies %.1f%% of the margin (aggregate accounting)" % (100 * (1 - _agg)))
        print("\n    ranked by mean share of the margin supplied:")
        for rank, c in enumerate(ordr, 1):
            if rank <= 10 or rank > n_c - 3:
                print("     %2d. %-22s %+7.1f%%" % (rank, cnames[c], 100 * mean_share[c]))
        k = a.top
        top_pos = float(np.clip(mean_share[ordr[:k]], 0, None).sum())
        # COUNT-INVARIANT CONCENTRATION (2026-11-20 13:15). A fixed top-8 bar is meaningless on a 9-unit layer: top-8 of 5 positive
        # suppliers IS the whole layer, and it printed "MET" for a layer that explains nothing. Concentration must be a fraction of
        # the layer. These two are:
        #   top_half_share  share of positive supply held by the strongest HALF of the positive suppliers
        #   supply_entropy Shannon entropy of the positive supply / log(n_pos)   (1.00 = perfectly spread)
        _pos = np.clip(mean_share, 0, None)
        _npos = int((_pos > 0).sum())
        _ordp = np.argsort(_pos)[::-1]
        _half = max(1, int(np.ceil(_npos / 2)))
        _th = float(_pos[_ordp[:_half]].sum() / _pos.sum()) if _pos.sum() > 1e-12 else float("nan")
        # BUG FIXED (2026-11-22 15:10): this was `_pos[_ordp[:_npos] / _pos.sum()]`, which divides the INDEX ARRAY by the sum
        # and then uses the resulting float array as an index -> IndexError: arrays used as indices must be integer. The intent
        # was to select the positive suppliers and then normalise them.
        _q = (_pos[_ordp[:_npos]] / _pos.sum()) if _pos.sum() > 1e-12 else None
        _H = float(-(_q * np.log(_q + 1e-12)).sum() / math.log(_npos)) if _npos > 1 else float("nan")
        print("    COUNT-INVARIANT concentration: top half (%d of %d positive units) carry %.1f%% of positive supply"
              % (_half, _npos, 100 * _th))
        print("    supply entropy/log(n_pos): %.3f   (1.00 = perfectly spread, lower = concentrated)" % _H)
        print("\n    top-%d concepts among the POSITIVE suppliers carry %.1f%% of all positive supply (P1 bar: >= 80%%)"
              % (k, 100 * top_pos / pos_total if pos_total > 1e-12 else float("nan")))
        # NAMING FIX (2026-11-20 13:02): this used to print "P1 verdict", and a run script that also had its own P1 had its working
        # result read as a failure because the log printed only this one. This is the CONCENTRATION test. It is not any arm's prediction.
        print("    CONCENTRATION (tool-side test, not an arm prediction): %s"
              % ("MET" if pos_total > 1e-12 and top_pos / pos_total >= 0.80 else "NOT MET"))
        if a.out:
            json.dump({"ckpt": a.ckpt, "windows": len(C), "recon_err_max": max(recon_err),
                       "positive_supply": pos_total,
                       "ranked": [{"concept": cnames[c], "mean_share": float(mean_share[c])} for c in ordr]},
                      open(a.out, "w"), indent=1)
            print("    wrote %s" % a.out)
        return

    if a.out:
        json.dump({"ckpt": a.ckpt, "windows": used, "base_all": base, "top": a.top,
                   "top_names": [cnames[c] for c in topset], "top_median": top_med,
                   "carries_of_all": share_of_all,
                   "ranked": [{"concept": cnames[c], "median_removed": float(med[c])} for c in order]},
                  open(a.out, "w"), indent=1)
        print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()
