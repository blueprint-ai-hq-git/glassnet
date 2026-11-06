#!/usr/bin/env python3
"""decode_step_loop_breakdown.py - the second pass of PLAN item (f): the breakdown must ADD UP.

The first pass (lens/scripts/decode_step_cost.py) timed each operation 200 times back to back and the
parts summed to only 24% (d=256) / 47% (d=512) of the step. That is a measurement of a different thing
(a hot micro-loop) than the step (a loop that slides a window, touches a 12 MB embedding table and two
multi-MB weight matrices per token). This pass times push and the step INSIDE the loop, with the load
average printed beside every arm, and then measures the shift itself - because the first pass said the
shift is the biggest part, which is a surprising enough claim to need its own measurement.

Predictions, written before the run: lens/runs/DECODE-STEP-COST-PREDICTIONS.txt (ADDENDUM 1, P46i-P46m).
Run:  bash lens/scripts/run_decode_step_loop_breakdown.sh
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import json                     # noqa: E402
import sys                      # noqa: E402
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

SHIPS = [("lens/runs/chat/cl_d256_attn_12_ep12.npz"),
         ("lens/runs/chat/sb_B1_d512_attn_ep96.npz")]
NSTEPS = 1500


def loop_split(net, stream, fused, do_push=True, reps=3):
    """One decoding loop, with push and the step timed SEPARATELY inside it. Median of `reps` runs."""
    W = int(net.W)
    best = None
    for _ in range(reps):
        c = WindowCache(net, fused=fused).seed(stream[:W])
        t_push = t_step = 0.0
        t0 = time.perf_counter()
        for t in stream[W:W + NSTEPS]:
            if do_push:
                a = time.perf_counter()
                c.push(t)
                b = time.perf_counter()
                cached_logits_h(net, c)
                e = time.perf_counter()
                t_push += b - a
                t_step += e - b
            else:
                b = time.perf_counter()
                cached_logits_h(net, c)
                e = time.perf_counter()
                t_step += e - b
        wall = time.perf_counter() - t0
        row = (t_push / NSTEPS, t_step / NSTEPS, wall / NSTEPS)
        if best is None or row[2] < best[2]:
            best = row
    return best                       # seconds per step: (push, step, wall)


def bench(fn, reps=300):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def main():
    print("=" * 100)
    print("CACHED DECODE STEP: the breakdown INSIDE the loop, and the shift measured on its own")
    print(f"pass started {time.strftime('%Y-%m-%d %H:%M:%S')}  OPENBLAS_NUM_THREADS="
          f"{os.environ.get('OPENBLAS_NUM_THREADS')}  numpy {np.__version__}")
    print("predictions (written first): lens/runs/DECODE-STEP-COST-PREDICTIONS.txt ADDENDUM 1 (P46i-P46m)")
    print("=" * 100)
    out = {"ships": {}, "host": {"numpy": np.__version__,
                                 "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS")}}
    verdicts = {}

    for ship in SHIPS:
        net = lens.load(os.path.join(ROOT, ship), device="cpu")
        name = os.path.basename(ship)
        W, D = int(net.W), int(net.D)
        rng = np.random.default_rng(0)
        stream = [int(t) for t in rng.integers(1, net.V - 2, size=W + NSTEPS + 10)]
        print(f"\n--- SHIP {name}  d={D} window={W}   loadavg now {tuple(round(x,2) for x in os.getloadavg())}")

        # ------------------------------------------------------------ P46i/P46j: inside the loop
        res = {}
        for fused in (False, True):
            loop_split(net, stream, fused)                     # warm up
            p_push, p_step, p_wall = loop_split(net, stream, fused)
            frac = (p_push + p_step) / p_wall
            arm = "reference (allocating)" if not fused else "fused (in place)"
            print(f"    {arm:<24s} push {p_push*1e6:7.1f} us   step {p_step*1e6:7.1f} us   "
                  f"wall {p_wall*1e6:7.1f} us   push+step = {frac*100:5.1f}% of the wall")
            res["reference" if not fused else "fused"] = {
                "push_us": p_push * 1e6, "step_us": p_step * 1e6, "wall_us": p_wall * 1e6,
                "push_plus_step_fraction": frac}
        _np, _ns, _nw = loop_split(net, stream, True, do_push=False)
        print(f"    fused, push DISABLED (the window never slides - wrong, isolates the shift): "
              f"step {_ns*1e6:7.1f} us   wall {_nw*1e6:7.1f} us")
        res["fused_no_push"] = {"step_us": _ns * 1e6, "wall_us": _nw * 1e6}
        fr = {k: v["push_plus_step_fraction"] for k, v in res.items() if "push_plus_step_fraction" in v}
        verdicts[f"P46i {name}"] = (min(fr.values()) >= 0.90,
                                    " ".join(f"{k} {v*100:.1f}%" for k, v in fr.items()))
        biggest = {k: ("push" if v["push_us"] > v["step_us"] else "step")
                   for k, v in res.items() if "push_us" in v}
        verdicts[f"P46j {name}"] = (all(b == "push" for b in biggest.values()),
                                    " ".join(f"{k}:{v}" for k, v in biggest.items()))

        # ------------------------------------------------------------ P46k/P46l: the shift itself
        f32 = np.float32
        Q = rng.standard_normal((W, D)).astype(f32)
        K = rng.standard_normal((W, D)).astype(f32)
        V = rng.standard_normal((W, D)).astype(f32)
        Vo = rng.standard_normal((W, D)).astype(f32)
        big = np.zeros((4, W, D), dtype=f32)
        big[0], big[1], big[2], big[3] = Q, K, V, Vo

        def shift4():
            Q[:-1] = Q[1:]
            K[:-1] = K[1:]
            V[:-1] = V[1:]
            Vo[:-1] = Vo[1:]

        def shift_combined():
            big[:, :-1] = big[:, 1:]

        # correctness of the combined shift, on a copy, before anything is concluded from its speed
        big2 = np.array([Q, K, V, Vo], dtype=f32)
        ref = np.array([np.roll(a, -1, axis=0) for a in (Q, K, V, Vo)], dtype=f32)
        big2[:, :-1] = big2[:, 1:]
        same = bool(np.array_equal(big2[:, :-1], ref[:, :-1]))

        def ring_write():
            big[:, 0] = Q[1]                       # the ring's only write: one row, no shift

        t4 = bench(shift4)
        tc = bench(shift_combined)
        tr = bench(ring_write)
        print(f"P46k the shift, measured alone ({4*(W-1)*D*4/1024:.0f} KB moved by the shipped one):")
        print(f"       four separate shifts (shipped)  {t4*1e6:8.1f} us")
        print(f"       one combined (4,W,D) shift      {tc*1e6:8.1f} us   ({tc/t4:.3f}x)   "
              f"bit-identical to the four: {same}")
        print(f"P46l ring write, no shift at all      {tr*1e6:8.1f} us   ({tr/t4:.3f}x the shipped shift)")
        verdicts[f"P46k {name}"] = (same and tc < t4, f"identical={same}, {tc/t4:.3f}x")
        verdicts[f"P46l {name}"] = (tr / t4 < 0.20, f"{tr/t4:.3f}x")
        res["shift"] = {"shift4_us": t4 * 1e6, "combined_us": tc * 1e6, "ring_write_us": tr * 1e6,
                        "combined_bit_identical": same}
        out["ships"][name] = {"d": D, "window": W, **res}

    print("\n" + "=" * 100)
    print("VERDICTS (binary; bars written before the run in DECODE-STEP-COST-PREDICTIONS.txt ADDENDUM 1)")
    for k, (ok, detail) in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}   [{detail}]")
    print(f"OVERALL: {'PASS' if all(o for o, _ in verdicts.values()) else 'FAIL'} "
          f"({sum(o for o, _ in verdicts.values())}/{len(verdicts)} parts)")
    out["verdicts"] = {k: bool(o) for k, (o, _) in verdicts.items()}
    out["overall"] = all(o for o, _ in verdicts.values())
    with open(os.path.join(ROOT, "lens/runs/decode_step_loop_breakdown.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("machine-readable numbers: lens/runs/decode_step_loop_breakdown.json")
    return 0 if out["overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
