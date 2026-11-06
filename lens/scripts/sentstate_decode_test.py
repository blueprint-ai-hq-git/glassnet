"""lens/scripts/sentstate_decode_test.py -- P115a-f: THE DECODE SIDE OF THE SENTSTATE GRAFT.

Predictions were written FIRST: lens/runs/SENTSTATE-DECODE-PREDICTIONS-P115.txt.
Output: lens/runs/SENTSTATE-DECODE.txt (append-only). CPU only, random-weight nets, no training, no GPU.

WHAT EACH BLOCK MEASURES, in one line each:
  P115a  the path an attention-OFF ship takes (make_cache -> None) IS net.forward, so the graft is exact;
  P115b  THE DEFECT: the KV cache's cached_logits never calls the model, so on a grafted ATTENTION ship it
         returns logits WITHOUT the state term -- measured against the same net's full path, with a base
         (un-grafted) net measured alongside as the float32 control;
  P115c  the fix (sentstate_wire.cached_logits_grafted) reproduces the full path on the same windows;
  P115d  lens/decode.py routes a grafted net to it and leaves the base path bit-identical;
  P115e  (NOT here -- it needs the stored checkpoint; lens/scripts/run_sentstate_decider_control.sh)
  P115f  api.load still drops the six tensors silently, re-measured on the REAL pre-flight checkpoint.

NOTHING HERE IS A QUALITY NUMBER. Random weights cannot talk and no talk, accuracy or ship claim is made.
"""
import json
import os
import sys
import time

import numpy as np

from lens import decode
from lens import kvcache
from lens import sentstate_wire as sw
from lens.concepts import Concept
from lens.model import LensNet

V, D, W, K, DS = 64, 32, 16, 5, 8
TINY_CKPT = "lens/runs/_tmp/ss_preflight/_ss_wire_on_ep1.npz"
OUT = "lens/runs/SENTSTATE-DECODE.txt"
_log_lines = []


def log(s=""):
    _log_lines.append(s)
    print(s, flush=True)


def flush_log(header):
    with open(OUT, "a") as f:
        f.write("\n" + "#" * 100 + "\n")
        f.write(header + "\n")
        f.write("#" * 100 + "\n")
        for l in _log_lines:
            f.write(l + "\n")


def concepts():
    return [Concept(f"c{i}", list(range(i * 4, i * 4 + 4)), support=0.25) for i in range(K)]


def make_net(attention, graft, seed=0):
    sw.install(ds=DS)
    if graft:
        sw.enable(ds=DS)
    else:
        sw.disable()
    try:
        return LensNet(V, concepts(), d=D, window=W, seed=seed, lam_concept=0.0, lam_act=0.0,
                       attention=attention)
    finally:
        sw.disable()          # leave the module's global flag as we found it


def full_logits(net, ctx):
    return np.asarray(net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"][0], dtype=np.float64)


def cache_for(net, ctx):
    c, why = decode.make_cache(net, ctx)
    return c, why


def op_count(net):
    return len(net.parameters()), sum(int(np.asarray(p.data).size) for p in net.parameters())


def verdict(name, ok, detail):
    log(f"  {name}: {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)


def main():
    t0 = time.time()
    rng = np.random.default_rng(1234)
    results = {}

    # ------------------------------------------------------------------ P115a
    log("=" * 100)
    log("P115a  THE FULL PATH IS THE ARM'S PATH (attention OFF -- the config every ship candidate here has)")
    net_off = make_net(attention=False, graft=True)
    ctx = [int(t) for t in rng.integers(0, V, size=W)]
    c, why = cache_for(net_off, ctx)
    a_same = (c is None) and ("NO attention" in why)
    log(f"    make_cache -> cache={c}  reason={why!r}")
    zs = decode.step_logits(net_off, ctx, None)
    zf = full_logits(net_off, ctx)
    dmax = float(np.max(np.abs(zs - zf)))
    results["P115a"] = verdict("P115a", a_same and dmax == 0.0,
                               f"cache is None with a named reason={a_same}   max|step_logits - forward| = {dmax:.3e}")
    n_off, p_off = op_count(net_off)
    log(f"    grafted attention-OFF net: {n_off} tensors / {p_off:,} params (the graft is ON the parameter list)")

    # ------------------------------------------------------------------ P115b
    log("")
    log("P115b  THE DEFECT: the KV cache is STATE-BLIND (attention ON)")
    net_g = make_net(attention=True, graft=True)
    net_b = make_net(attention=True, graft=False)      # the float32 control: same architecture, no graft
    ctxs = [[int(t) for t in rng.integers(0, V, size=W)] for _ in range(8)]
    d_base, d_graft, base_vs_full = [], [], []
    top1_full, top1_cache = [], []
    for cx in ctxs:
        zf = full_logits(net_g, cx)
        cg, _ = cache_for(net_g, cx)
        zg = np.asarray(kvcache.cached_logits(net_g, cg), dtype=np.float64)
        d_graft.append(float(np.max(np.abs(zg - zf))))
        top1_full.append(int(np.argmax(zf)))
        top1_cache.append(int(np.argmax(zg)))
        zb = np.asarray(kvcache.cached_logits(net_b, cache_for(net_b, cx)[0]), dtype=np.float64)
        base_vs_full.append(float(np.max(np.abs(zb - full_logits(net_b, cx)))))
    log(f"    CONTROL (un-grafted attention net), cache vs its own full path: max|d| = {max(base_vs_full):.3e} "
        f"over {len(ctxs)} windows   <- this is what float32 noise looks like here")
    log(f"    GRAFTED net, cache (kvcache.cached_logits) vs its own full path: max|d| = {max(d_graft):.6f} "
        f"   median {float(np.median(d_graft)):.6f}")
    log(f"    top-1 token from the state-blind cache agrees with the full path on "
        f"{sum(1 for a, b in zip(top1_full, top1_cache) if a == b)}/{len(ctxs)} windows")
    results["P115b"] = verdict("P115b", max(d_graft) > 0.5 and max(base_vs_full) < 1e-4,
                               f"grafted max|d| {max(d_graft):.6f} > 0.5 AND control {max(base_vs_full):.3e} < 1e-4")

    # ------------------------------------------------------------------ P115c
    log("")
    log("P115c  THE GRAFTED CACHED STEP REPRODUCES THE FULL PATH")
    d_fix, top1_fix = [], []
    for cx in ctxs:
        zf = full_logits(net_g, cx)
        cg, _ = cache_for(net_g, cx)
        zx = np.asarray(sw.cached_logits_grafted(net_g, cg)[0], dtype=np.float64)
        d_fix.append(float(np.max(np.abs(zx - zf))))
        top1_fix.append(int(np.argmax(zx)) == int(np.argmax(zf)))
    log(f"    grafted cached step vs full path: max|d| = {max(d_fix):.6f}   median {float(np.median(d_fix)):.6f}")
    log(f"    top-1 agreement: {sum(top1_fix)}/{len(ctxs)} windows")
    results["P115c"] = verdict("P115c", max(d_fix) <= 5e-2 and all(top1_fix),
                               f"max|d| {max(d_fix):.6f} <= 5e-2 AND top-1 {sum(top1_fix)}/{len(ctxs)}")

    # ------------------------------------------------------------------ P115d
    log("")
    log("P115d  THE SHIPPED DECODER USES IT, AND THE BASE PATH IS UNCHANGED")
    ok_d = True
    dmax_base = 0.0
    for cx in ctxs[:4]:
        cg, _ = cache_for(net_g, cx)
        z_dec = decode.step_logits(net_g, cx, cg)
        z_ref = np.asarray(sw.cached_logits_grafted(net_g, cg)[0], dtype=np.float64)
        ok_d = ok_d and float(np.max(np.abs(z_dec - z_ref))) == 0.0
        cb, _ = cache_for(net_b, cx)
        z_dec_b = decode.step_logits(net_b, cx, cb)
        z_ref_b = np.asarray(kvcache.cached_logits(net_b, cb), dtype=np.float64)
        dmax_base = max(dmax_base, float(np.max(np.abs(z_dec_b - z_ref_b))))
    z_dec_b2, h_dec_b2 = decode.step_logits_and_concepts(net_b, ctxs[0], cache_for(net_b, ctxs[0])[0])
    z_ref_b2, h_ref_b2 = kvcache.cached_logits_h(net_b, cache_for(net_b, ctxs[0])[0])
    ok_d = ok_d and float(np.max(np.abs(z_dec_b2 - np.asarray(z_ref_b2, np.float64)))) == 0.0
    ok_d = ok_d and float(np.max(np.abs(h_dec_b2 - np.asarray(h_ref_b2, np.float32)))) == 0.0
    log(f"    grafted: decode.step_logits == cached_logits_grafted to the bit: {ok_d}")
    log(f"    base net: decode.step_logits vs kvcache.cached_logits (the PRE-EDIT implementation) "
        f"max|d| = {dmax_base:.3e}; step_logits_and_concepts returns the same (z, h) bit for bit")
    results["P115d"] = verdict("P115d", ok_d and dmax_base == 0.0,
                              f"graft routing bit-exact={ok_d} AND base path bit-identical={dmax_base == 0.0}")

    # ------------------------------------------------------------------ P115f (P115e lives in its own runner)
    log("")
    log("P115f  api.load IS STILL THE UNSAFE LOADER (re-measured on the REAL pre-flight checkpoint)")
    if not os.path.exists(TINY_CKPT):
        log(f"    SKIPPED: {TINY_CKPT} does not exist")
        results["P115f"] = None
    else:
        z = np.load(TINY_CKPT, allow_pickle=False)
        keys = set(z.files) - {"__meta__"}
        present = [k for k in sw.GRAFT_PARAMS if k in keys]
        net_gr = sw.load_grafted(TINY_CKPT)
        net_ap = __import__("lens.api", fromlist=["api"]).load(TINY_CKPT)
        w_g = int(np.asarray(net_gr.W))
        Vg = int(net_gr.V)
        ctx_real = [int(t) for t in np.random.default_rng(7).integers(0, Vg, size=w_g)]
        zg_ = np.asarray(net_gr.forward_arrays(np.asarray([ctx_real], dtype=np.int64))["logits"][0], np.float64)
        za_ = np.asarray(net_ap.forward_arrays(np.asarray([ctx_real], dtype=np.int64))["logits"][0], np.float64)
        dmax_real = float(np.max(np.abs(zg_ - za_)))
        log(f"    checkpoint carries {len(present)}/{len(sw.GRAFT_PARAMS)} graft tensors: {sorted(present)}")
        log(f"    load_grafted: {op_count(net_gr)[0]} tensors   api.load: {op_count(net_ap)[0]} tensors "
            f"(difference {op_count(net_gr)[1] - op_count(net_ap)[1]:,} params)")
        log(f"    max|logit(load_grafted) - logit(api.load)| = {dmax_real:.6f}   (no exception raised by api.load)")
        results["P115f"] = verdict("P115f", len(present) == len(sw.GRAFT_PARAMS) and dmax_real > 0.1,
                                   f"{len(present)}/6 tensors present AND max|d| {dmax_real:.6f} > 0.1")

    log("")
    log("=" * 100)
    log("SUMMARY: " + "  ".join(f"{k}={'SKIP' if v is None else ('PASS' if v else 'FAIL')}"
                                for k, v in results.items()))
    log(f"elapsed {time.time() - t0:.1f} s   interpretation: WIRING ONLY -- random weights, no training, "
        f"no talk or accuracy claim, nothing shipped.")
    log("=" * 100)
    flush_log(f"=== SENTSTATE DECODE TEST (P115a-f) {time.strftime('%F %H:%M:%S')}   "
              f"predictions: lens/runs/SENTSTATE-DECODE-PREDICTIONS-P115.txt")
    bad = [k for k, v in results.items() if v is False]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
