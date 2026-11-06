#!/usr/bin/env python3
"""opt_step_cost.py -- PLAN item (d2c): ATTRIBUTE the per-epoch cost of our own step rule.

WHY THIS EXISTS (read the observation first, it is in lens/runs/OPT-VS-SGD.txt and in the
`seconds` field of lens/runs/chat/*_history.jsonl): on the SAME corpus, SAME seed, SAME 12
epochs and SAME 70,320 steps, only the step rule differing,
    opt_sgd04_s0   67.7 s/epoch        opt_relgrad05_s0  247.4 s/epoch
so ~180 s/epoch, ~30.7 ms per step, is unaccounted for. Those arms did NOT share machine
load, so that number is a SIGNAL TO ATTRIBUTE, not a finding. This script attributes it.

WHAT IS AND IS NOT FAITHFUL HERE, STATED BEFORE THE NUMBERS:
  FAITHFUL: the architecture is the library's own (lens.Net) at the real d=256 / W=32 /
    V=12,002 / K=36 shapes, VERIFIED against the checkpoint that actually ran
    (lens/runs/chat/opt_relgrad05_s0_ep12.npz); the tensors are the real Tensor objects
    from net.parameters(); the step rule is the shipped lens/opt.py RelativeGrad; the decay
    factor is the one the real run printed (0.999944369820945); both backends are the real
    ones (lens.backend.use_gpu).
  NOT FAITHFUL, DECLARED: the gradients are a seeded random draw SCALED to the norms the
    real probe recorded (lens/runs/OPT-GRAD-SCALE.txt), not the backward() of a real batch.
    The cost of a norm, a cast and a copy is data-independent, and P45e TESTS that claim
    (two draws) instead of asserting it.

NOTHING IN lens/opt.py IS CHANGED BY THIS SCRIPT. The 16-arm comparison in
lens/runs/QUEUE.txt is in flight and must differ between arms in the step rule and nothing
else; a float32 norm is not bit-identical to a float64 one, so the candidate speedup (arm C)
is measured here and written up, not shipped.

Predictions P45a-g were written BEFORE this ran: lens/runs/OPT-STEP-COST-PREDICTIONS.txt.
Output: lens/runs/OPT-STEP-COST.txt (+ opt_step_cost.json).
"""
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.

import lens                                                             # noqa: E402
import lens.tensor as T                                                 # noqa: E402
from lens import backend as B                                           # noqa: E402
from lens.concepts import Concept                                       # noqa: E402
from lens.opt import RelativeGrad                                       # noqa: E402

OUT = os.path.join(ROOT, "lens", "runs", "OPT-STEP-COST.txt")
JSONOUT = os.path.join(ROOT, "lens", "runs", "opt_step_cost.json")
CKPT = os.path.join(ROOT, "lens", "runs", "chat", "opt_relgrad05_s0_ep12.npz")
DECAY = 0.999944369820945           # the real per-step factor the relgrad05 s0 run printed
V, K, D, W = 12002, 36, 256, 32
# the norms the REAL probe measured on one real batch (lens/runs/OPT-GRAD-SCALE.txt)
REAL_NORMS = {"emb": 4.5841e-02, "concept.W": 1.8319e-01, "concept.b": 4.3256e-02,
              "answer.W": 3.3160e-01, "answer.b": 1.1031e-01}
STEPS_PER_EPOCH = 5860              # 70,320 opt_steps / 12 epochs, the relgrad05 s0 run
GAP_S = 247.4 - 67.7                # relgrad05 s0 minus sgd04 s0, s/epoch, same seed/config
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def sync():
    if B.is_gpu():
        import cupy
        cupy.cuda.Stream.null.synchronize()


def timeit(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return ts


def med_ms(ts):
    return statistics.median(ts) * 1e3


def build(device):
    """The real model, on `device`, with real parameter tensors. Shapes are checked against
    the checkpoint that actually ran, so a wrong d/K/V cannot pass silently."""
    T.set_device(device)
    B.use_gpu(device == "gpu")
    concepts = [Concept(f"synth_{i:02d}", [i + 1]) for i in range(K)]
    net = lens.Net(vocab=V, concepts=concepts, d=D, window=W, seed=0,
                   lam_concept=0.3, lam_act=0.0, device=device)
    return net


def set_grads(params, gseed):
    """Gradients whose norms are EXACTLY the real probe's, drawn from a fixed seed."""
    rng = np.random.default_rng(gseed)
    out = {}
    for p in params:
        if p.name not in REAL_NORMS:
            continue
        g = rng.standard_normal(p.data.shape).astype(np.float32)
        g *= np.float32(REAL_NORMS[p.name] / float(np.linalg.norm(g)))
        p.grad = B.to_dev(g)
        out[p.name] = float(np.linalg.norm(np.asarray(B.to_cpu(p.grad), dtype=np.float64)))
    return out


def clear_grads(params):
    for p in params:
        p.grad = None


def plain_step(params, lr):
    """train_chat.py's own rule, device-resident: p.data = p.data - lr * p.grad."""
    for p in params:
        if p.grad is not None:
            p.data = p.data - lr * p.grad


def device_relgrad_step(params, lr, eps=1e-8):
    """CANDIDATE (arm C, NOT shipped): the same rule, no host round-trip at all.
    norm on the device, divide on the device, subtract on the device."""
    xp = B.xp
    for p in params:
        if p.grad is None:
            continue
        denom = xp.linalg.norm(p.grad) + eps
        p.data = p.data - (lr * p.grad / denom)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class _Pad:
    """A fake parameter with 10x the embedding's entries. TIMING CONTROL ONLY (P45f): if the
    shipped rule is not ~10x slower on 10x the work, the harness is timing Python overhead."""

    name = "emb_x10"

    def __init__(self, src, mult=10):
        self.data = B.xp.concatenate([src.data] * mult, axis=0)
        self.grad = B.xp.concatenate([src.grad] * mult, axis=0)


def pad_control(params, lr, reps):
    emb = [p for p in params if p.name == "emb"]
    if not emb:
        return None, None
    opt_real = RelativeGrad(emb, lr=lr, mode="relgrad", decay=DECAY)
    real_ms = med_ms(timeit(opt_real.step, reps))
    opt_pad = RelativeGrad([_Pad(emb[0])], lr=lr, mode="relgrad", decay=DECAY)
    pad_ms = med_ms(timeit(opt_pad.step, 5))
    return real_ms, pad_ms


def main():
    t_start = time.strftime("%F %H:%M:%S")
    opt_sha_before = sha(os.path.join(ROOT, "lens", "opt.py"))
    load0 = os.getloadavg()
    say("=" * 96)
    say(f"OPT-STEP-COST.txt -- PLAN item (d2c), ATTRIBUTION of our optimiser's per-epoch cost.")
    say(f"start {t_start}   host load average {load0[0]:.2f} {load0[1]:.2f} {load0[2]:.2f}")
    say("predictions written BEFORE this ran: lens/runs/OPT-STEP-COST-PREDICTIONS.txt (P45a-g)")
    say("CALIBRATION ONLY. No accuracy claim of any kind is made here, and nothing is trained.")
    say("=" * 96)
    running = os.popen("pgrep -cf '[t]rain_chat'").read().strip()
    say(f"  concurrent train_chat processes at start: {running}   (recorded because GPU timing")
    say(f"  under another job's load is a ratio-within-this-process, not an absolute)")
    say("")

    # ---------------------------------------------------------------- shapes verified
    z = np.load(CKPT, allow_pickle=True)
    say("SHAPES: the model used here against the checkpoint that actually ran")
    say(f"  {os.path.basename(CKPT)}")
    net_cpu = build("cpu")
    shape_rows = []
    for p in net_cpu.parameters():
        want = z[p.name].shape if p.name in z.files else None
        got = tuple(p.data.shape)
        ok = want is not None and tuple(want) == got
        shape_rows.append((p.name, got, want, ok))
        say(f"    {p.name:<12} built {str(got):<16} checkpoint {str(tuple(want)) if want is not None else '-':<16}"
            f" {'MATCH' if ok else 'MISMATCH'}")
    if not all(r[3] for r in shape_rows):
        raise SystemExit("REFUSED: the built model does not match the checkpoint that ran.")
    say("  -> the architecture is the real one at the real shapes; a wrong d/K/V cannot pass.")
    say("")

    results = {}
    # ================================================================ CPU backend
    say("=" * 96)
    say("CPU BACKEND (lens.backend.use_gpu(False)) -- separates ARITHMETIC from TRANSFER")
    say("=" * 96)
    params = net_cpu.parameters()
    achieved = set_grads(params, 1)
    say("  gradients set to the REAL probe's norms (lens/runs/OPT-GRAD-SCALE.txt):")
    for p in params:
        say(f"    {p.name:<12} n {p.data.size:>9,}   ||g|| here {achieved[p.name]:.6e}   "
            f"real {REAL_NORMS[p.name]:.6e}   rel err {abs(achieved[p.name]-REAL_NORMS[p.name])/REAL_NORMS[p.name]:.2e}")
    opt = RelativeGrad(params, lr=0.05, mode="relgrad", decay=DECAY)
    reps = 30
    tA = timeit(opt.step, reps)
    tB = timeit(lambda: plain_step(params, 0.4), reps)
    tC = timeit(lambda: device_relgrad_step(params, 0.05), reps)
    cpu_real_ms, cpu_pad_ms = pad_control(params, 0.05, reps)
    say(f"  A shipped opt.step()          median {med_ms(tA):9.3f} ms  (n={reps})  "
        f"min {min(tA)*1e3:.3f}  max {max(tA)*1e3:.3f}")
    say(f"  B plain device update (SGD)   median {med_ms(tB):9.3f} ms  (n={reps})")
    say(f"  C device-resident relgrad     median {med_ms(tC):9.3f} ms  (n={reps})")
    r_cpu_ab = med_ms(tA) / med_ms(tB)
    r_cpu_ac = med_ms(tA) / med_ms(tC)
    say(f"  ratio A/B (P45d bar >= 2.0)   {r_cpu_ab:.3f}x")
    say(f"  ratio A/C                     {r_cpu_ac:.3f}x")
    say(f"  D CONTROL (cpu): shipped step on 10x-padded emb {cpu_pad_ms:.3f} ms vs {cpu_real_ms:.3f} ms "
        f"-> {cpu_pad_ms/cpu_real_ms:.3f}x")
    results["cpu"] = {"A_ms": med_ms(tA), "B_ms": med_ms(tB), "C_ms": med_ms(tC),
                      "A_over_B": r_cpu_ab, "A_over_C": r_cpu_ac, "reps": reps,
                      "pad_real_ms": cpu_real_ms, "pad_pad_ms": cpu_pad_ms}

    # the defining property of the shipped rule must survive arm C
    say("")
    say("  PROPERTY CHECK on arm C (the shipped rule's defining property, decay=1.0):")
    p0 = {p.name: np.array(B.to_cpu(p.data), copy=True) for p in params}
    device_relgrad_step(params, 0.05)
    worst = 0.0
    for p in params:
        d = np.asarray(B.to_cpu(p.data), dtype=np.float64) - p0[p.name].astype(np.float64)
        nrm = float(np.linalg.norm(d))
        rel = abs(nrm - 0.05) / 0.05
        worst = max(worst, rel)
        say(f"    {p.name:<12} ||update|| {nrm:.8f}   lr 0.05   rel err {rel:.3e}")
    say(f"    worst relative error {worst:.3e}   (P45b bar: <= 1e-5) -> "
        f"{'PASS' if worst <= 1e-5 else 'FAIL'}")
    results["C_property_worst_rel"] = worst
    # restore real gradients for the GPU phase
    clear_grads(params)

    # ================================================================ GPU backend
    say("")
    say("=" * 96)
    say("GPU BACKEND (lens.backend.use_gpu(True)) -- what the 16 arms actually run")
    say("=" * 96)
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
    except Exception as e:                                    # noqa: BLE001
        import traceback
        say(f"  NO GPU AVAILABLE ({type(e).__name__}: {e}) -- the GPU half of this item is NOT measured.")
        say("  traceback (recorded because a broken GPU import is a REAL BUG, not a footnote):")
        for ln in traceback.format_exc().splitlines():
            say(f"    {ln}")
        results["gpu"] = None
    else:
        net_gpu = build("gpu")
        gparams = net_gpu.parameters()
        achieved_g = set_grads(gparams, 1)
        optg = RelativeGrad(gparams, lr=0.05, mode="relgrad", decay=DECAY)
        say(f"  gradients: same real norms, same seed; emb ||g|| here {achieved_g['emb']:.6e} "
            f"vs real {REAL_NORMS['emb']:.6e}")
        reps_g = 30
        gA = timeit(optg.step, reps_g)
        gB = timeit(lambda: plain_step(gparams, 0.4), reps_g)
        gC = timeit(lambda: device_relgrad_step(gparams, 0.05), reps_g)
        say(f"  A shipped opt.step()          median {med_ms(gA):9.3f} ms  (n={reps_g})  "
            f"min {min(gA)*1e3:.3f}  max {max(gA)*1e3:.3f}")
        say(f"  B plain device update (SGD)   median {med_ms(gB):9.3f} ms  (n={reps_g})")
        say(f"  C device-resident relgrad     median {med_ms(gC):9.3f} ms  (n={reps_g})")
        r_gpu_ab = med_ms(gA) / med_ms(gB)
        r_gpu_ac = med_ms(gA) / med_ms(gC)
        say(f"  ratio A/B (P45a bar >= 4.0)   {r_gpu_ab:.3f}x")
        say(f"  ratio A/C (P45b bar >= 3.0)   {r_gpu_ac:.3f}x")

        # where the cost sits: the embedding alone vs everything else
        emb = [p for p in gparams if p.name == "emb"]
        rest = [p for p in gparams if p.name != "emb"]
        opt_emb = RelativeGrad(emb, lr=0.05, mode="relgrad", decay=DECAY)
        opt_rest = RelativeGrad(rest, lr=0.05, mode="relgrad", decay=DECAY)
        eA = timeit(opt_emb.step, reps_g)
        eB = timeit(lambda: plain_step(emb, 0.4), reps_g)
        rA = timeit(opt_rest.step, reps_g)
        rB = timeit(lambda: plain_step(rest, 0.4), reps_g)
        say(f"    emb   (3,072,512 entries): shipped {med_ms(eA):8.3f} ms   plain {med_ms(eB):7.3f} ms   "
            f"ratio {med_ms(eA)/med_ms(eB):6.2f}x")
        say(f"    other (  739,022 entries): shipped {med_ms(rA):8.3f} ms   plain {med_ms(rB):7.3f} ms   "
            f"ratio {med_ms(rA)/med_ms(rB):6.2f}x")

        # CONTROL (P45f): the shipped rule on a tensor 10x bigger must cost ~10x more
        pad = [p for p in gparams if p.name == "emb"][0]
        dReal, dPad = pad_control(gparams, 0.05, reps_g)
        say(f"  D CONTROL: shipped step on 10x-padded emb  median {dPad:9.3f} ms "
            f"vs {dReal:.3f} ms -> {dPad/dReal:.3f}x  (P45f bar >= 3.0x)")

        # transfer counts + bandwidth
        cnt = {"to_cpu": 0, "to_dev": 0, "bytes_cpu": 0, "bytes_dev": 0}
        o_cpu, o_dev = B.to_cpu, B.to_dev

        def w_cpu(a, _o=o_cpu):
            cnt["to_cpu"] += 1
            cnt["bytes_cpu"] += int(getattr(a, "nbytes", 0))
            return _o(a)

        def w_dev(a, _o=o_dev):
            cnt["to_dev"] += 1
            cnt["bytes_dev"] += int(getattr(a, "nbytes", 0))
            return _o(a)

        B.to_cpu, B.to_dev = w_cpu, w_dev
        try:
            for k in cnt:
                cnt[k] = 0
            optg.step()
            shipped_xfers = dict(cnt)
            for k in cnt:
                cnt[k] = 0
            plain_step(gparams, 0.4)
            plain_xfers = dict(cnt)
            for k in cnt:
                cnt[k] = 0
            device_relgrad_step(gparams, 0.05)
            cand_xfers = dict(cnt)
        finally:
            B.to_cpu, B.to_dev = o_cpu, o_dev
        say("")
        say("  HOST<->DEVICE TRAFFIC PER STEP (counted, not inferred):")
        say(f"    shipped opt.step()      to_cpu {shipped_xfers['to_cpu']:>3} calls "
            f"({shipped_xfers['bytes_cpu']/1e6:7.2f} MB)   to_dev {shipped_xfers['to_dev']:>3} calls "
            f"({shipped_xfers['bytes_dev']/1e6:7.2f} MB)")
        say(f"    plain device update     to_cpu {plain_xfers['to_cpu']:>3} calls "
            f"({plain_xfers['bytes_cpu']/1e6:7.2f} MB)   to_dev {plain_xfers['to_dev']:>3} calls "
            f"({plain_xfers['bytes_dev']/1e6:7.2f} MB)")
        say(f"    device-resident relgrad to_cpu {cand_xfers['to_cpu']:>3} calls "
            f"({cand_xfers['bytes_cpu']/1e6:7.2f} MB)   to_dev {cand_xfers['to_dev']:>3} calls "
            f"({cand_xfers['bytes_dev']/1e6:7.2f} MB)")

        host = np.asarray(B.to_cpu(pad.data))
        dev = B.to_dev(host)
        h2d = timeit(lambda: B.to_dev(host), 20)
        d2h = timeit(lambda: B.to_cpu(dev), 20)
        mb = host.nbytes / 1e6
        say("")
        say(f"  RAW BANDWIDTH on this machine (12.29 MB float32 = the embedding, 20 reps):")
        say(f"    host->device {med_ms(h2d):8.3f} ms  = {mb/(med_ms(h2d)/1e3):7.2f} MB/s")
        say(f"    device->host {med_ms(d2h):8.3f} ms  = {mb/(med_ms(d2h)/1e3):7.2f} MB/s")

        # implied epoch share (P45c)
        implied = STEPS_PER_EPOCH * (med_ms(gA) - med_ms(gB)) / 1e3
        say("")
        say(f"  IMPLIED EPOCH SHARE (P45c): {STEPS_PER_EPOCH:,} steps x ({med_ms(gA):.3f} - "
            f"{med_ms(gB):.3f}) ms = {implied:.1f} s/epoch")
        say(f"    measured same-seed gap (relgrad05 s0 {247.4:.1f} s - sgd04 s0 {67.7:.1f} s, "
            f"lens/runs/chat/*_history.jsonl) = {GAP_S:.1f} s/epoch  -> ratio {implied/GAP_S:.3f}x")
        results["gpu"] = {"A_ms": med_ms(gA), "B_ms": med_ms(gB), "C_ms": med_ms(gC),
                          "A_over_B": r_gpu_ab, "A_over_C": r_gpu_ac, "reps": reps_g,
                          "emb_A_ms": dReal, "emb_B_ms": med_ms(eB),
                          "rest_A_ms": med_ms(rA), "rest_B_ms": med_ms(rB),
                          "pad_D_ms": dPad, "shipped_xfers": shipped_xfers,
                          "plain_xfers": plain_xfers, "cand_xfers": cand_xfers,
                          "h2d_ms": med_ms(h2d), "d2h_ms": med_ms(d2h),
                          "implied_epoch_s": implied, "measured_gap_s": GAP_S}

        # P45e: a second gradient draw must not decide the timing
        set_grads(gparams, 2)
        gA2 = timeit(optg.step, 15)
        gB2 = timeit(lambda: plain_step(gparams, 0.4), 15)
        dA = abs(med_ms(gA2) - med_ms(gA)) / med_ms(gA)
        dB = abs(med_ms(gB2) - med_ms(gB)) / med_ms(gB)
        say("")
        say(f"  P45e second draw (seed 2): A {med_ms(gA2):.3f} ms vs {med_ms(gA):.3f} ms "
            f"({dA*100:.1f}% apart); B {med_ms(gB2):.3f} ms vs {med_ms(gB):.3f} ms ({dB*100:.1f}% apart)")
        say(f"    bar: both < 10% -> {'PASS' if max(dA, dB) < 0.10 else 'FAIL'}")
        results["P45e"] = {"A_drift": dA, "B_drift": dB}

    # ---------------------------------------------------------------- verdicts
    say("")
    say("=" * 96)
    say("VERDICTS -- each printed beside the values it compares")
    say("=" * 96)
    g = results.get("gpu")
    if g:
        v = "PASS" if g["A_over_B"] >= 4.0 else ("FAIL" if g["A_over_B"] < 2.0 else "FAIL (between 2x and 4x)")
        say(f"  P45a  shipped step / plain update on GPU = {g['A_over_B']:.3f}x   bar >= 4.0   -> {v}")
        say(f"        (cpu {results['cpu']['A_over_B']:.3f}x, bar >= 2.0 -> "
            f"{'PASS' if results['cpu']['A_over_B'] >= 2.0 else 'FAIL'})   [P45d]")
        xf = g["shipped_xfers"]
        ok_b = (xf["to_cpu"] >= 4 and xf["to_dev"] >= 4 and g["A_over_C"] >= 3.0
                and results["C_property_worst_rel"] <= 1e-5)
        say(f"  P45b  shipped step does {xf['to_cpu']} to_cpu + {xf['to_dev']} to_dev calls per step "
            f"({(xf['bytes_cpu']+xf['bytes_dev'])/1e6:.2f} MB round-tripped); "
            f"device-resident is {g['A_over_C']:.3f}x faster (bar >= 3.0); "
            f"its update norm is exact to {results['C_property_worst_rel']:.2e} (bar <= 1e-5)")
        say(f"        -> {'PASS' if ok_b else 'FAIL'}")
        r = g["implied_epoch_s"] / GAP_S
        say(f"  P45c  implied {g['implied_epoch_s']:.1f} s/epoch vs measured {GAP_S:.1f} s/epoch "
            f"= {r:.3f}x   bar 0.5x-2.0x   -> {'PASS' if 0.5 <= r <= 2.0 else 'FAIL'}")
        say(f"  P45f  control, 10x-padded emb {g['pad_D_ms']:.3f} ms vs real {g['emb_A_ms']:.3f} ms "
            f"= {g['pad_D_ms']/g['emb_A_ms']:.3f}x   bar >= 3.0   -> "
            f"{'PASS' if g['pad_D_ms']/g['emb_A_ms'] >= 3.0 else 'FAIL'}")
        say(f"  P45e  two gradient draws: A {results['P45e']['A_drift']*100:.1f}%, "
            f"B {results['P45e']['B_drift']*100:.1f}% apart   bar < 10%   -> "
            f"{'PASS' if max(results['P45e'].values()) < 0.10 else 'FAIL'}")
    else:
        say("  P45a/P45b/P45c/P45e/P45f: NOT MEASURED (no GPU in this process).")
    opt_sha_after = sha(os.path.join(ROOT, "lens", "opt.py"))
    say(f"  P45g  lens/opt.py sha256 unchanged: {opt_sha_before == opt_sha_after}   "
        f"({opt_sha_before[:16]}...)   -> {'PASS' if opt_sha_before == opt_sha_after else 'FAIL'}")
    load1 = os.getloadavg()
    running1 = os.popen("pgrep -cf '[t]rain_chat'").read().strip()
    say("")
    say(f"  host load average at end {load1[0]:.2f} {load1[1]:.2f} {load1[2]:.2f}; "
        f"concurrent train_chat at end: {running1}")
    say("  NOT CLAIMED: no accuracy claim, no training result, nothing about which step rule")
    say("  learns better. This file contains a COST measurement only.")
    say("=" * 96)
    with open(OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(JSONOUT, "w") as fh:
        json.dump(results, fh, indent=1, sort_keys=True)
    print(f"\nwrote {OUT}\nwrote {JSONOUT}", flush=True)


if __name__ == "__main__":
    main()
