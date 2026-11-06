#!/usr/bin/env python3
"""decode_push_breakdown.py - third pass of PLAN item (f): WHAT inside push costs 88-315 us?

Established by lens/scripts/decode_step_loop_breakdown.py, measured in place and bit-identical:
  d=256  push 88.1 us of a 298.3 us step,  d=512  push 315.1 us of a 744.1 us step
  the window SHIFT alone is 9.1 us (d=256) and 13.3 us (d=512) - so the slide is not the cost.
This script times every sub-part of push IN PLACE (emb row fetch, the four separate projections), then
the same push with the three input projections batched into ONE matvec, and finally the whole loop under
OPENBLAS_NUM_THREADS=1 versus 3.

Predictions, written before the run: lens/runs/DECODE-STEP-COST-PREDICTIONS.txt (ADDENDUM 2, P46n-P46s).
Run:  bash lens/scripts/run_decode_push_breakdown.sh 1     (and again with 3)
"""
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = sys.argv[1] if len(sys.argv) > 1 else "3"

import json                     # noqa: E402
import time                     # noqa: E402

import numpy as np              # noqa: E402

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
# REMOVED 2026-10-20, THE SAME REAL BUG IN A FORM THE FIRST GATE COULD NOT SEE (class 3;
# evidence lens/runs/SYSPATH-SHADOW-FIX-class3.txt, probe lens/runs/GPU-ATTN-CACHE.txt): this
# tuple put <project>/lens on sys.path as well, so the top-level name `cuda` resolved to our own
# lens/cuda/ package and cupy 14.2 died with "ModuleNotFoundError: No module named
# 'cuda.pathfinder'". lens/ is imported as `lens.X` via the ROOT entry and lens/scripts/ has its
# own entry, so the `/lens` element is not needed. Do not add it back.
for _p in (ROOT, ROOT + "/lens/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lens                                             # noqa: E402
from lens.kvcache import WindowCache, cached_logits_h   # noqa: E402

SHIPS = ["lens/runs/chat/cl_d256_attn_12_ep12.npz", "lens/runs/chat/sb_B1_d512_attn_ep96.npz"]
NSTEPS = 1200
THREADS = os.environ["OPENBLAS_NUM_THREADS"]
pc = time.perf_counter


def push_parts(net, stream):
    """The shipped push, split into its parts, timed in place, in the shipped order."""
    W = int(net.W)
    c = WindowCache(net, fused=True).seed(stream[:W])
    e = sh = pq = pk = pv = po = 0.0
    t0 = pc()
    for t in stream[W:W + NSTEPS]:
        a = pc(); x = c.emb[int(t)]; b = pc(); e += b - a
        a = pc()
        for buf in (c.Q, c.K, c.V, c.Vo):
            buf[:-1] = buf[1:]
        c.ids[:-1] = c.ids[1:]
        c.ids[-1] = int(t)
        b = pc(); sh += b - a
        a = pc(); c.Q[-1] = x @ c.Wq; b = pc(); pq += b - a
        a = pc(); c.K[-1] = x @ c.Wk; b = pc(); pk += b - a
        a = pc(); c.V[-1] = x @ c.Wv; b = pc(); pv += b - a
        a = pc(); c.Vo[-1] = c.V[-1] @ c.Wo; b = pc(); po += b - a
    return {"emb": e, "shift": sh, "proj_q": pq, "proj_k": pk, "proj_v": pv, "proj_o": po,
            "wall": pc() - t0}


def push_batched(net, stream):
    """The same push with x @ [Wq|Wk|Wv] done as ONE matvec, and a bit-identity check against the three."""
    W = int(net.W)
    c = WindowCache(net, fused=True).seed(stream[:W])
    Wcat = np.concatenate([c.Wq, c.Wk, c.Wv], axis=1)       # (D, 3D)
    D = int(net.D)
    identical = True
    tb = 0.0
    t0 = pc()
    for t in stream[W:W + NSTEPS]:
        x = c.emb[int(t)]
        a = pc()
        y = x @ Wcat
        c.Q[-1] = y[:D]
        c.K[-1] = y[D:2 * D]
        c.V[-1] = y[2 * D:]
        c.Vo[-1] = c.V[-1] @ c.Wo
        b = pc(); tb += b - a
        for buf in (c.Q, c.K, c.V, c.Vo):
            buf[:-1] = buf[1:]
        c.ids[:-1] = c.ids[1:]
        c.ids[-1] = int(t)
    # bit-identity of the batched projections against the three separate ones, on a fresh window
    c2 = WindowCache(net, fused=True).seed(stream[:W])
    xs = c2.emb[np.asarray(stream[W:W + 50], dtype=np.int64)]
    one = xs @ Wcat
    three = np.concatenate([xs @ c2.Wq, xs @ c2.Wk, xs @ c2.Wv], axis=1)
    identical = bool(np.array_equal(one, three))
    return {"batched_us_per_step": tb / NSTEPS, "wall": pc() - t0, "bit_identical": identical}


def full_loop(net, stream, fused=True):
    W = int(net.W)
    c = WindowCache(net, fused=fused).seed(stream[:W])
    t0 = pc()
    for t in stream[W:W + NSTEPS]:
        cached_logits_h(net, c)
        c.push(t)
    return (pc() - t0) / NSTEPS


def main():
    print("=" * 100)
    print(f"PUSH BREAKDOWN + BATCHED PROJECTIONS + THREAD COUNT   OPENBLAS_NUM_THREADS={THREADS}")
    print(f"pass started {time.strftime('%Y-%m-%d %H:%M:%S')}  numpy {np.__version__}  "
          f"loadavg {tuple(round(x, 2) for x in os.getloadavg())}")
    print("predictions (written first): lens/runs/DECODE-STEP-COST-PREDICTIONS.txt ADDENDUM 2 (P46n-P46s)")
    print("=" * 100)
    out = {"threads": THREADS, "ships": {}}
    verdicts = {}

    for ship in SHIPS:
        net = lens.load(os.path.join(ROOT, ship), device="cpu")
        name = os.path.basename(ship)
        W, D = int(net.W), int(net.D)
        rng = np.random.default_rng(0)
        stream = [int(t) for t in rng.integers(1, net.V - 2, size=W + NSTEPS + 60)]
        push_parts(net, stream)                      # warm
        r = push_parts(net, stream)
        parts = r["emb"] + r["shift"] + r["proj_q"] + r["proj_k"] + r["proj_v"] + r["proj_o"]
        projs = r["proj_q"] + r["proj_k"] + r["proj_v"] + r["proj_o"]
        # the shipped push, measured as one call in the same conditions
        c = WindowCache(net, fused=True).seed(stream[:W])
        t0 = pc()
        for t in stream[W:W + NSTEPS]:
            c.push(t)
        shipped = (pc() - t0) / NSTEPS
        # BUG FIXED HERE (found by reading the output): the first version divided a TOTAL over NSTEPS by a
        # PER-STEP time, so every fraction came out NSTEPS times too big (157,538% of push). The printed
        # percentages were right, the VERDICTS were nonsense. Both now divide by NSTEPS.
        parts_ps, projs_ps, shift_ps = parts / NSTEPS, projs / NSTEPS, r["shift"] / NSTEPS
        frac = parts_ps / shipped
        print(f"\n--- {name}  d={D} window={W}   push measured as ONE call: {shipped*1e6:.1f} us")
        for label, v in (("emb row fetch (random row of a 12 MB table)", r["emb"]),
                         ("the window shift (5 buffers)", r["shift"]),
                         ("x @ Wq", r["proj_q"]), ("x @ Wk", r["proj_k"]),
                         ("x @ Wv", r["proj_v"]), ("V[-1] @ Wo", r["proj_o"])):
            print(f"       {label:<46s} {v/NSTEPS*1e6:8.1f} us   {v/NSTEPS/shipped*100:5.1f}% of push")
        print(f"       {'SUM OF SUB-PARTS':<46s} {parts_ps*1e6:8.1f} us   {frac*100:5.1f}%   "
              f"(bar >= 85%)")
        print(f"       {'the four projections together':<46s} {projs_ps*1e6:8.1f} us   "
              f"{projs_ps/shipped*100:5.1f}%   (bar >= 50%)")
        print(f"       {'the shift alone':<46s} {shift_ps*1e6:8.1f} us   "
              f"{shift_ps/shipped*100:5.1f}%   (bar < 15%)")
        verdicts[f"P46n {name}"] = (frac >= 0.85, f"{frac*100:.1f}% of push")
        verdicts[f"P46o {name}"] = (projs_ps / shipped >= 0.50, f"{projs_ps/shipped*100:.1f}% of push")
        verdicts[f"P46p {name}"] = (shift_ps / shipped < 0.15, f"{shift_ps/shipped*100:.1f}% of push")

        b = push_batched(net, stream)
        print(f"P46q batched [Wq|Wk|Wv] in ONE matvec: {b['batched_us_per_step']*1e6:.1f} us for the "
              f"projections+Vo against {projs_ps*1e6:.1f} us separately "
              f"({b['batched_us_per_step']/projs_ps:.3f}x)   bit-identical: {b['bit_identical']}")
        verdicts[f"P46q {name}"] = (b["bit_identical"] and b["batched_us_per_step"] < projs_ps,
                                    f"identical={b['bit_identical']}, "
                                    f"{b['batched_us_per_step']/projs_ps:.3f}x")

        full_loop(net, stream)
        wall = full_loop(net, stream)
        print(f"P46s the whole loop (push + step) with {THREADS} BLAS thread(s): {wall*1e6:.1f} us/step")
        # ---------------------------------------------------------------- P46t/P46u: is it BYTES, not calls?
        # Every weight tensor a decoded token must read, from the SHAPES of this ship - not estimated.
        traf = {"emb row": D * 4,
                "Wq+Wk+Wv+Wo": 4 * D * D * 4,
                "concept.W": W * D * int(net.K) * 4,
                "concept.b": int(net.K) * 4,
                "answer.W": int(net.K) * int(net.V) * 4,
                "answer.b": int(net.V) * 4,
                "Q,K,V,Vo window": 4 * W * D * 4}
        tot = sum(traf.values())
        gbs = tot / wall / 1e9
        print(f"P46t weight traffic per decoded token, from the shapes: {tot/1e6:.2f} MB")
        for k, v in traf.items():
            print(f"       {k:<18s} {v/1e6:8.2f} MB   {v/tot*100:5.1f}%")
        print(f"P46u achieved bandwidth over the whole loop: {tot/1e6:.2f} MB / {wall*1e6:.1f} us = "
              f"{gbs:.1f} GB/s   (bar: 8-20 GB/s means BANDWIDTH-bound, not call-overhead-bound)")
        verdicts[f"P46t {name}"] = (3.5e6 <= tot <= 1.0e7, f"{tot/1e6:.2f} MB/token")
        verdicts[f"P46u {name}"] = (8.0 <= gbs <= 20.0, f"{gbs:.1f} GB/s")
        out["ships"][name] = {"d": D, "push_us": shipped * 1e6, "subparts_us": {k: v / NSTEPS * 1e6
                                                                               for k, v in r.items()
                                                                               if k != "wall"},
                              "subparts_fraction": frac, "projections_fraction": projs_ps / shipped,
                              "shift_fraction": shift_ps / shipped,
                              "batched_us": b["batched_us_per_step"] * 1e6,
                              "batched_bit_identical": b["bit_identical"],
                              "loop_us_per_step": wall * 1e6, "traffic_bytes_per_token": tot,
                              "traffic_mb": tot / 1e6, "achieved_gbs": gbs,
                              "traffic_parts": traf}

    print("\n" + "=" * 100)
    print(f"VERDICTS at {THREADS} thread(s) (binary; bars written before the run)")
    for k, (ok, detail) in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}   [{detail}]")
    out["verdicts"] = {k: bool(o) for k, (o, _) in verdicts.items()}
    with open(os.path.join(ROOT, f"lens/runs/decode_push_breakdown_t{THREADS}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"machine-readable numbers: lens/runs/decode_push_breakdown_t{THREADS}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
