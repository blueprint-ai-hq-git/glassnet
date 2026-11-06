#!/usr/bin/env python3
"""f16_gemv_bench.py - PLAN item (f2b) part 1: does OUR OWN fused f16 GEMV beat numpy's float32 GEMV?

Predictions written FIRST: lens/runs/F16-GEMV-PREDICTIONS.txt (P49a-e are judged here; P49f-i are judged
by lens/scripts/f16_gemv_test.py, which needs the two ships and is queued so it cannot overlap a training
arm - two arms have already been truncated by this machine's 14 GB ceiling).

LIGHT ON PURPOSE: it reads weight shapes and values straight out of the stored .npz files and never builds
a model, so it can run while a training arm is in flight.

Writes lens/runs/F16-GEMV-BENCH.txt (and lens/runs/f16_gemv_bench.json).
Run:  .venv-gpu/bin/python3 lens/scripts/f16_gemv_bench.py
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import json                      # noqa: E402
import sys                       # noqa: E402
import time                      # noqa: E402

import numpy as np               # noqa: E402

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
for _p in (ROOT, ROOT + "/lens/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lens.fusedgemv import gemv_singleton      # noqa: E402

SHIPS = [("lens/runs/chat/cl_d256_attn_12_ep12.npz", 256),
         ("lens/runs/chat/sb_B1_d512_attn_ep96.npz", 512)]
MATS = ["attn.Wq", "attn.Wk", "attn.Wv", "attn.Wo", "concept.W", "answer.W"]
# VARIANT -> thread counts measured. "generic" is the first version (input-major, #pragma omp simd); it
# measured 4.5-9 GB/s and is kept on the record. "avx2" is the block-outer register-accumulator kernel.
VARIANTS = [("generic", [1, 3]), ("avx2", [1, 3, 12])]
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def med_us(fn, rng, warm=10):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(rng):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(ts))


def main():
    g = gemv_singleton()
    say("=" * 100)
    say("PLAN ITEM (f2b) PART 1 - OUR OWN FUSED f16 GEMV vs NUMPY, micro-benchmark on the REAL shapes")
    say(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}   numpy {np.__version__}   "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}   cpu_count={g.nt_max}")
    say(f"build: {' '.join(g.cmd)}")
    say(f"load average at start: {os.getloadavg()}  (a training arm is in flight on this machine, so the")
    say("  ABSOLUTE times below are inflated and not stable to 10%; the RATIOS are within one process at")
    say("  one instant and are what the verdicts use - the same caveat recorded in OPT-STEP-COST.txt)")
    say("predictions (written first): lens/runs/F16-GEMV-PREDICTIONS.txt   (P49a-e here)")
    say("=" * 100)

    out = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "numpy": np.__version__,
           "threads_env": os.environ.get("OPENBLAS_NUM_THREADS"), "cpu_count": g.nt_max,
           "build_cmd": g.cmd, "ships": {}}

    # ------------------------------------------------------------------ P49b the conversion, ALL 65536 patterns
    say("\nP49b f16 -> f32 CONVERSION - EVERY ONE of the 65,536 bit patterns, both code paths:")
    bits = np.arange(65536, dtype=np.uint16)
    ref = bits.view(np.float16).astype(np.float32)              # numpy's own cast
    fast = g.decode(bits, "fast")
    sw = g.decode(bits, "sw")
    conv = {}
    for name, got in (("fast (F16C/_Float16)", fast), ("software", sw)):
        both_nan = np.isnan(ref) & np.isnan(got)
        exact = (ref.view(np.uint32) == got.view(np.uint32)) | both_nan
        bad = int((~exact).sum())
        nan_mismatch = int((np.isnan(ref) ^ np.isnan(got)).sum())
        conv[name] = {"mismatching_patterns": bad, "nan_vs_number_mismatch": nan_mismatch}
        say(f"    {name:22s} mismatching patterns {bad:6d} of 65536   "
            f"nan-vs-number {nan_mismatch:6d}   -> {'PASS' if bad == 0 else 'FAIL'}")
    say(f"    subnormals checked: {int(((bits & 0x7C00) == 0).sum())} patterns, "
        f"inf/nan patterns {int(((bits & 0x7C00) == 0x7C00).sum())}")
    out["conversion"] = conv

    # ------------------------------------------------------------------ per-matrix micro-benchmark
    for ship, d in SHIPS:
        p = os.path.join(ROOT, ship)
        z = np.load(p, allow_pickle=False)
        name = os.path.basename(ship)
        say(f"\n--- SHIP {name}  d={d}")
        rec = {"d": d, "matrices": {}}
        out["ships"][name] = rec
        tot = {"numpy_f32": 0.0, "numpy_f16_widen": 0.0}
        tot_k = {(v, t): 0.0 for v, ts in VARIANTS for t in ts}
        tot_bytes = {"f32": 0, "f16": 0}
        for key in MATS:
            W = np.ascontiguousarray(z[key], dtype=np.float32)
            n_in, n_out = W.shape
            W16 = W.astype(np.float16)
            b16 = np.ascontiguousarray(W16).view(np.uint16)
            rng = np.random.default_rng(20261021)
            x = rng.standard_normal((1, n_in)).astype(np.float32)
            rng_t = 200 if W.size <= 600000 else 60

            t32 = med_us(lambda: x @ W, rng_t)
            t16w = med_us(lambda: x @ W16.astype(np.float32), rng_t)
            tk = {}
            for v, ts in VARIANTS:
                for t in ts:
                    tk[(v, t)] = med_us(lambda v=v, t=t: g.gemv(b16, x, nthreads=t, variant=v), rng_t)
            # correctness against numpy ON THE SAME f16 VALUES (P49c), for BOTH variants
            ref16 = (x @ W16.astype(np.float32)).astype(np.float64)
            diffs = {}
            for v in ("generic", "avx2"):
                got = g.gemv(b16, x, nthreads=3, variant=v).astype(np.float64)
                diffs[v] = float(np.abs(ref16 - got).max())
            scale = float(np.abs(ref16).max())
            # the same f16 weights through the deliberately slow reference kernel (small ones only)
            if W.size <= 200000:
                sc = g.gemv(b16, x, scalar=True).astype(np.float64)
                dsc = float(np.abs(sc - g.gemv(b16, x, nthreads=3, variant="avx2").astype(np.float64)).max())
            else:
                dsc = float("nan")
            f32b, f16b = int(W.size * 4), int(W.size * 2)
            say(f"    {key:11s} ({n_in:6d},{n_out:6d})  f32 {f32b/1e6:6.3f} MB  f16 {f16b/1e6:6.3f} MB")
            say(f"        numpy f32 GEMV      {t32:9.2f} us  ({f32b/t32/1e3:6.2f} GB/s read)")
            say(f"        numpy f16->f32 then  {t16w:9.2f} us  ({t16w/max(t32,1e-9):6.3f}x vs f32)   "
                f"<- the (f2) losing arm")
            for v, ts in VARIANTS:
                for t in ts:
                    say(f"        {v:7s} {t:2d} threads  {tk[(v, t)]:9.2f} us  "
                        f"({t32/max(tk[(v, t)],1e-9):6.3f}x vs numpy f32, {f16b/tk[(v, t)]/1e3:6.2f} GB/s read)")
            say(f"        correctness: max|kernel-numpy| generic {diffs['generic']:.6e} / "
                f"avx2 {diffs['avx2']:.6e} on a scale of {scale:.4e}   | scalar vs avx2 {dsc:.6e}")
            rec["matrices"][key] = {"shape": [n_in, n_out], "f32_bytes": f32b, "f16_bytes": f16b,
                                    "numpy_f32_us": t32, "numpy_f16_widen_us": t16w,
                                    "kernel_us": {f"{v}@{t}": tk[(v, t)] for v, ts in VARIANTS for t in ts},
                                    "max_abs_diff_vs_numpy": diffs, "ref_scale": scale,
                                    "scalar_vs_avx2_max_diff": dsc}
            tot["numpy_f32"] += t32
            tot["numpy_f16_widen"] += t16w
            for v, ts in VARIANTS:
                for t in ts:
                    tot_k[(v, t)] += tk[(v, t)]
            tot_bytes["f32"] += f32b
            tot_bytes["f16"] += f16b

        say(f"\n    WEIGHT PART OF ONE DECODED TOKEN (sum of the six matrices, ONE pass each):")
        say(f"        float32 bytes {tot_bytes['f32']:,}   f16 bytes {tot_bytes['f16']:,}   "
            f"ratio {tot_bytes['f32']/tot_bytes['f16']:.4f}x fewer  (P49a bar: exactly 2.0000x)")
        say(f"        numpy f32 GEMV     {tot['numpy_f32']:9.2f} us/token")
        say(f"        numpy f16 widened  {tot['numpy_f16_widen']:9.2f} us/token  "
            f"({tot['numpy_f16_widen']/tot['numpy_f32']:.3f}x vs f32)")
        for v, ts in VARIANTS:
            for t in ts:
                say(f"        {v:7s} {t:2d} threads   {tot_k[(v, t)]:9.2f} us/token  "
                    f"({tot_k[(v, t)]/tot['numpy_f32']:.3f}x vs numpy f32, i.e. "
                    f"{tot['numpy_f32']/max(tot_k[(v, t)],1e-9):.3f}x faster)")
        rec["token_totals_us"] = {"numpy_f32": tot["numpy_f32"],
                                  "numpy_f16_widen": tot["numpy_f16_widen"],
                                  "kernel": {f"{v}@{t}": tot_k[(v, t)] for v, ts in VARIANTS for t in ts},
                                  "bytes": tot_bytes}
        # ---- P49d / P49e verdicts, with the compared values beside them
        aw = rec["matrices"]["answer.W"]
        cw = rec["matrices"]["concept.W"]
        k = aw["kernel_us"]["avx2@3"]
        kc = cw["kernel_us"]["avx2@3"]
        kg = aw["kernel_us"]["generic@3"]
        say(f"\n    P49d answer.W  {aw['numpy_f32_us']:.2f} us numpy vs {k:.2f} us avx2@3t = "
            f"{aw['numpy_f32_us']/max(k,1e-9):.3f}x   (bar 1.3-2.2x)  -> "
            f"{'PASS' if 1.3 <= aw['numpy_f32_us']/max(k,1e-9) <= 2.2 else 'FAIL'}"
            f"   [the first kernel, generic@3t, was {aw['numpy_f32_us']/max(kg,1e-9):.3f}x]")
        say(f"    P49e concept.W {cw['numpy_f32_us']:.2f} us numpy vs {kc:.2f} us avx2@3t = "
            f"{cw['numpy_f32_us']/max(kc,1e-9):.3f}x   (bar 1.2-2.0x)  -> "
            f"{'PASS' if 1.2 <= cw['numpy_f32_us']/max(kc,1e-9) <= 2.0 else 'FAIL'}")

    say("\n" + "=" * 100)
    say("PART 2 (the real loop, bytes/token, ms/step, token identity on BOTH ships, and the control) is")
    say("lens/scripts/f16_gemv_test.py -> lens/runs/F16-GEMV.txt, queued in lens/runs/QUEUE.txt so it")
    say("cannot overlap a training arm. P49f-i are judged there.")
    say("=" * 100)
    with open(os.path.join(ROOT, "lens/runs/F16-GEMV-BENCH.txt"), "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(ROOT, "lens/runs/f16_gemv_bench.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote lens/runs/F16-GEMV-BENCH.txt + lens/runs/f16_gemv_bench.json")


if __name__ == "__main__":
    main()
