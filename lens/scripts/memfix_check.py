#!/usr/bin/env python3
"""memfix_check.py - DOES THE CHUNKED GATHER RETURN EXACTLY WHAT THE OLD ONE RETURNED?

Written 2026-10-22 for predictions P62a/P62b/P62f (lens/runs/MEMFIX-PREDICTIONS.txt).
The old implementation is loaded from the pre-change copy lens/runs/_tmp/chat_data_prefix_memfix.py, the new
one from lens/scripts/chat_data.py, and both are called on the SAME ids/bounds/split/seed. Everything is
compared value by value: shape, dtype, every element of X and Y, and the returned window starts.

It also prints the size of the transient index array each implementation needs at the full-corpus window
count (P62b) - arithmetic on measured counts, not a guess.

Run:  .venv-gpu/bin/python3 -u lens/scripts/memfix_check.py
Never writes anything except stdout; no corpus is touched.
"""
import importlib.util
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OLD = os.path.join(ROOT, "lens", "runs", "_tmp", "chat_data_prefix_memfix.py")
NEW = os.path.join(ROOT, "lens", "scripts", "chat_data.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_corpus(n_conv=60, vmax=12001, seed=7, lo=40, hi=700):
    rng = np.random.default_rng(seed)
    lens = rng.integers(lo, hi, size=n_conv)
    bounds = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    ids = rng.integers(0, vmax, size=int(bounds[-1])).astype(np.int32)
    split = np.zeros(n_conv, dtype=np.uint8)
    split[rng.choice(n_conv, size=max(1, n_conv // 5), replace=False)] = 1
    return ids, bounds, split


def main():
    old = load(OLD, "chat_data_old_memfix")
    new = load(NEW, "chat_data_new_memfix")
    ids, bounds, split = make_corpus()
    print(f"corpus: {len(ids):,} tokens, {len(bounds)-1:,} conversations, "
          f"train windows would be {sum(max(0, int(bounds[j+1]-bounds[j])-32-1)//16 for j in range(len(bounds)-1)):,}")
    fails = []
    cases = []
    for window in (32, 17):
        for cap in (None, 1000, 4_000_000):
            for seed in (0, 3):
                cases.append((window, cap, seed))
    rows = []
    for window, cap, seed in cases:
        kw = dict(max_windows=cap, seed=seed)
        try:
            Xo, Yo, So = old.make_windows(ids, bounds, split, window, which=0, return_index=True, **kw)
            Xn, Yn, Sn = new.make_windows(ids, bounds, split, window, which=0, return_index=True, **kw)
        except Exception as e:                                     # noqa: BLE001
            fails.append(f"window={window} cap={cap} seed={seed}: raised {type(e).__name__}: {e}")
            continue
        ok = (Xo.shape == Xn.shape and Yo.shape == Yo.shape and So.shape == Sn.shape
              and Xo.dtype == Xn.dtype and Yo.dtype == Yn.dtype
              and np.array_equal(Xo, Xn) and np.array_equal(Yo, Yn) and np.array_equal(So, Sn))
        if not ok:
            maxd = int(np.abs(Xo.astype(np.int64) - Xn.astype(np.int64)).max()) if Xo.shape == Xn.shape else -1
            fails.append(f"window={window} cap={cap} seed={seed}: MISMATCH shape {Xo.shape} vs {Xn.shape} "
                         f"dtype {Xo.dtype}/{Xn.dtype}, max|dX|={maxd}")
        rows.append((window, cap, seed, Xo.shape, str(Xo.dtype), ok))
    print(f"\nCASES ({len(rows)} of {len(cases)}):")
    for window, cap, seed, shape, dtype, ok in rows:
        print(f"  window={window:<3} max_windows={str(cap):<9} seed={seed}  X{shape} {dtype}  identical={ok}")
    print(f"\nP62a {'PASS' if not fails else 'FAIL'}: identical on {sum(1 for r in rows if r[-1])}/{len(cases)} cases")
    for f in fails:
        print("   " + f)

    # ---- P62b: transient index array at the FULL corpus window count --------------------------------
    N_FULL = 16_878_132
    window = 32
    old_bytes = N_FULL * window * 8                     # the old code's idx2d, int64, at once
    cells = 1 << 23                                     # the same bound the new code uses
    step = max(1, cells // window)
    new_bytes = min(N_FULL, step) * window * 8
    print(f"\nP62b transient index array at N={N_FULL:,}, window={window}:")
    print(f"   old: {old_bytes/1e9:.2f} GB at once      new: {new_bytes/1e6:.1f} MB (chunk {step:,} windows)")
    print(f"   result X itself: {N_FULL*window*8/1e9:.2f} GB in BOTH (dtype int64, unchanged)")
    print(f"NS = no verdict on accuracy: this file measures equality and bytes only.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
