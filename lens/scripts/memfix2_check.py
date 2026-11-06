#!/usr/bin/env python3
"""memfix2_check.py - the WIDTH fix: do the values move, and what does it save?

Written 2026-10-22 16:4x for predictions P64a and P64e (lens/runs/MEMFIX2-PREDICTIONS.txt, written BEFORE the
edit). The pre-change implementation is loaded from lens/runs/_tmp/chat_data_prefix_memfix2.py, the changed one
from lens/scripts/chat_data.py, and both are called on the SAME ids/bounds/split/seed.

WHAT IS COMPARED, and why the comparison is this way:
  * SHAPE and every ELEMENT of X, Y and the window starts, compared after casting both sides to int64. The
    dtype is what changed ON PURPOSE (int64 -> int32), so comparing dtypes would only restate the change;
    comparing VALUES is the claim ("the values are untouched").
  * The changed function MUST return int32 - if it silently returned int64 the memory prediction is false
    and this check says FAIL rather than passing on a stale assumption.
  * The REAL corpus is then read in chunks and its maximum token id is printed: int32 can only be safe if
    every id is far below 2^31 (P64e).
  * The byte arithmetic on the full window count is printed from MEASURED shapes, not estimated.

Run:  .venv-gpu/bin/python3 -u lens/scripts/memfix2_check.py
Never writes anything but stdout; no stored file is touched.
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OLD = os.path.join(ROOT, "lens", "runs", "_tmp", "chat_data_prefix_memfix2.py")
NEW = os.path.join(ROOT, "lens", "scripts", "chat_data.py")
CORPUS = os.path.join(ROOT, "data", "chat", "prepared_mix_ctrl_honest")


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
    ids = rng.integers(0, vmax, size=int(bounds[-1])).astype(np.int64)
    split = np.zeros(n_conv, dtype=np.uint8)
    split[rng.choice(n_conv, size=max(1, n_conv // 5), replace=False)] = 1
    return ids, bounds, split


def main():
    old = load(OLD, "chat_data_old_memfix2")
    new = load(NEW, "chat_data_new_memfix2")
    ids, bounds, split = make_corpus()
    print(f"corpus: {len(ids):,} tokens, {len(bounds)-1:,} conversations")
    fails, rows = [], []
    for window in (32, 17):
        for cap in (None, 1000, 4_000_000):
            for seed in (0, 3):
                kw = dict(max_windows=cap, seed=seed)
                try:
                    Xo, Yo, So = old.make_windows(ids, bounds, split, window, which=0,
                                                  return_index=True, **kw)
                    Xn, Yn, Sn = new.make_windows(ids, bounds, split, window, which=0,
                                                  return_index=True, **kw)
                except Exception as e:                                     # noqa: BLE001
                    fails.append(f"window={window} cap={cap} seed={seed}: raised {type(e).__name__}: {e}")
                    continue
                same_vals = (Xo.shape == Xn.shape and Yo.shape == Yn.shape and So.shape == Sn.shape
                             and np.array_equal(Xo.astype(np.int64), Xn.astype(np.int64))
                             and np.array_equal(Yo.astype(np.int64), Yn.astype(np.int64))
                             and np.array_equal(So, Sn))
                width_ok = Xn.dtype == np.int32 and Yn.dtype == np.int32
                if not same_vals:
                    fails.append(f"window={window} cap={cap} seed={seed}: VALUES MOVED "
                                 f"shape {Xo.shape} vs {Xn.shape}")
                if not width_ok:
                    fails.append(f"window={window} cap={cap} seed={seed}: dtype is {Xn.dtype}/{Yn.dtype}, "
                                 f"NOT int32 - the memory prediction P64d is then false")
                rows.append((window, cap, seed, Xo.shape, str(Xo.dtype), str(Xn.dtype), same_vals, width_ok))
    print(f"\nCASES ({len(rows)}):")
    for window, cap, seed, shape, do_, dn, sv, wo in rows:
        print(f"  window={window:<3} cap={str(cap):<9} seed={seed}  X{shape}  old {do_} -> new {dn}  "
              f"values_identical={sv}  width_int32={wo}")
    val_pass = sum(1 for r in rows if r[6])
    wid_pass = sum(1 for r in rows if r[7])
    print(f"\nP64a {'PASS' if not fails and val_pass == len(rows) else 'FAIL'}: "
          f"values identical on {val_pass}/{len(rows)} cases (dtype changed to int32 on {wid_pass}/{len(rows)})")
    for f in fails:
        print("   " + f)

    # ---- P64e: the real corpus' token ids, in chunks, so this never needs the whole file in RAM ----------
    f_ids = os.path.join(CORPUS, "ids_v12000.npy")
    if os.path.exists(f_ids):
        on_disk = np.load(f_ids, mmap_mode="r")
        mx, mn = 0, 1 << 40
        step = 1 << 24
        for s in range(0, len(on_disk), step):
            chunk = np.asarray(on_disk[s:s + step])
            mx = max(mx, int(chunk.max()))
            mn = min(mn, int(chunk.min()))
        d = np.load(f_ids, mmap_mode="r")
        print(f"\nP64e REAL CORPUS (file {os.path.relpath(f_ids, ROOT)}):")
        print(f"   on-disk dtype {d.dtype}, {len(d):,} tokens, id range {mn}..{mx}, int32 max "
              f"{np.iinfo(np.int32).max:,}")
        ok32 = mx < np.iinfo(np.int32).max
        print(f"   P64e {'PASS' if ok32 else 'FAIL'}: every id fits int32 with "
              f"{'enormous' if ok32 else 'NO'} headroom -> int32 cannot truncate a token id")
    else:
        print(f"\nP64e SKIPPED: no {f_ids} (nothing claimed)")

    # ---- P64d: the byte arithmetic, from the measured window count and window width --------------------
    N_CAPPED, N_FULL, window = 6_000_000, 16_878_132, 32
    for name, N in (("capped arm", N_CAPPED), ("FULL CORPUS", N_FULL)):
        X32, X64 = N * window * 4, N * window * 8
        Y32, Y64 = N * 4, N * 8
        print(f"   {name:>11} N={N:,}: X {X64/1e9:.2f} GB -> {X32/1e9:.2f} GB   "
              f"Y {Y64/1e9:.3f} -> {Y32/1e9:.3f} GB   saved {(X64-X32+Y64-Y32)/1e9:.2f} GB")
    print(f"P64d (the run's own peak RSS) is NOT measured here - it is read from the arm's scope")
    print(f"   (systemctl --user show -p MemoryPeak glassnet-q-<unit>.scope) when it runs.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
