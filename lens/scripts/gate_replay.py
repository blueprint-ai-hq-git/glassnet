#!/usr/bin/env python3
"""gate_replay.py — OFFLINE REPLAY of the shipped memorisation-gate rule on one arm's own per-step log.

WHAT IT IS: a text-only replayer. It opens ONE per-step log
(lens/runs/chat/gate_sgd04_s2_gates.jsonl, written by lens/scripts/train_chat.py --gates gap), replays the
shipped rule (GapGate variant 'ema_band', beta 0.5) over the arm's OWN logged (train_loss, probe_loss) series
and compares the replayed verdict with the verdict the arm recorded. Then it replays the same series under
alternative commit policies and tolerances.

WHAT IT IS NOT: no model, no corpus, no GPU, no training, no engine edit, nothing written into any run. It does
not claim that a differently-gated run would produce these numbers: under another policy the weights — and so
the later losses — would differ. The numbers are conditional on the logged trajectory, as predicted (P109i).

THE MECHANISM IT TESTS (read from the code, not assumed):
  train_chat.py 1191-1203 : candidate step is applied, the probe is re-scored, gate.evaluate() decides, and a
                            refused step is REVERTED, and gate.commit() is called ONLY when the step is accepted.
  unified.py GapGate.commit : "Apply the step's EMAs. Only accepted steps move the state."
  unified.py GapGate.decide (ema_band): refuse  <=>  cand_h > ema_held*(1+tol),  cand_h = b*ema_held + (1-b)*p
     with tol=0 that is exactly  refuse <=> p > ema_held , i.e. "refuse unless this step's probe loss is at or
     below the reference", and because the reference is committed only on accepted steps it is a running lower
     envelope of the probe loss.

USAGE: python3 lens/scripts/gate_replay.py  [--file PATH] [--beta 0.5]
Predictions: lens/runs/GATE-ARM-GRADE-PREDICTIONS-P109.txt (written before this file existed).
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
DEFAULT_LOG = os.path.join(PROJ, "lens/runs/chat/gate_sgd04_s2_gates.jsonl")


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def replay(rows, *, ref_policy, tol, variant, beta):
    """Replay one policy over the logged series. Returns (refused_flags, committed_held, committed_train).

    ref_policy: 'on_accept'   = the shipped behaviour: the EMA state moves only on accepted steps
                'every_step'  = the fix under test: the EMA is the EMA of the series on every step
    variant:    'ema_band'  = refuse <=> cand_h > ref_h*(1+tol)
                'gap_level' = refuse <=> cand_t < ref_h*(1-tol)   (detector 3a, the memorisation signature)
    """
    refused = []
    committed_h = []
    committed_t = []
    ref_t = None
    ref_h = None
    accepted_seen = 0
    for r in rows:
        p = float(r["probe_loss"])
        t = float(r["train_loss"])
        if ref_h is None:                      # first step: nothing to compare against, always accepted
            cand_t, cand_h = t, p
            refuse = False
        else:
            cand_t = beta * ref_t + (1.0 - beta) * t
            cand_h = beta * ref_h + (1.0 - beta) * p
            if variant == "ema_band":
                refuse = bool(cand_h > ref_h * (1.0 + tol))
            else:
                refuse = bool(cand_t < ref_h * (1.0 - tol))
        refused.append(refuse)
        if (not refuse) or ref_policy == "every_step":
            ref_t, ref_h = cand_t, cand_h
            committed_h.append(cand_h)
            committed_t.append(cand_t)
        accepted_seen += 1
    return refused, committed_h, committed_t


def frac(flags):
    return (sum(1 for f in flags if f) / len(flags)) if flags else float("nan")


def per_epoch(rows, flags):
    out = OrderedDict()
    for r, f in zip(rows, flags):
        ep = int(r["epoch"])
        d = out.setdefault(ep, [0, 0])
        d[0] += 1
        d[1] += int(f)
    return out


def log(msg=""):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_LOG)
    ap.add_argument("--beta", type=float, default=0.5)
    a = ap.parse_args()

    rows = load(a.file)
    n = len(rows)
    logged_refused = [bool(r["refused"]) for r in rows]
    steps = [int(r["step"]) for r in rows]
    epochs = sorted({int(r["epoch"]) for r in rows})

    me = os.path.relpath(os.path.abspath(__file__), PROJ)
    log("=" * 110)
    log(f"GATE REPLAY — OFFLINE, TEXT ONLY   {time.strftime('%F %H:%M:%S')}   [{me}]")
    log("predictions P109a-i: lens/runs/GATE-ARM-GRADE-PREDICTIONS-P109.txt (written before this file existed)")
    log(f"input log (read-only): {a.file}")
    log(f"beta = {a.beta:g}   rows = {n:,}   steps {min(steps)}..{max(steps)}   epochs {epochs}")
    log("=" * 110)

    # ---------- 1. contiguity / readability (P59d, on what exists) ----------
    contiguous = steps == list(range(min(steps), max(steps) + 1))
    log("")
    log("--- 1. THE ARM'S OWN LOG (P59d, on the rows that exist) ---")
    log(f"  rows {n:,}; one row per step, steps contiguous 1..{max(steps)}: {contiguous}")
    log(f"  logged refused total {sum(logged_refused):,}/{n:,} = {frac(logged_refused):.4f}")
    for ep, (tot, ref) in per_epoch(rows, logged_refused).items():
        log(f"    epoch {ep}: refused {ref:,}/{tot:,} = {ref / tot:.4f}"
            + ("   (epoch incomplete)" if tot < 5860 else ""))
    log("  P59d needs 70,320 rows for 12 epochs of 5,860 steps: only decidable at ep12 — this is the partial read.")

    # ---------- 2. the equivalence test (P109a) and the ratchet (P109b) ----------
    log("")
    log("--- 2. THE SHIPPED RULE, REPLAYED (commit ONLY on accepted steps, tol=0) — P109a / P109b ---")
    ref_as_shipped, comm_h, comm_t = replay(rows, ref_policy="on_accept", tol=0.0, variant="ema_band",
                                            beta=a.beta)
    mism = [i for i in range(n) if ref_as_shipped[i] != logged_refused[i]]
    log(f"  replayed refusals {sum(ref_as_shipped):,}/{n:,} = {frac(ref_as_shipped):.4f}")
    log(f"  the arm's own     {sum(logged_refused):,}/{n:,} = {frac(logged_refused):.4f}")
    log(f"  MISMATCHES vs the arm's own flag: {len(mism)} of {n:,}"
        + (f"   first at step {steps[mism[0]]}" if mism else "   -> P109a PASS (the rule model is exact)"))
    inc = sum(1 for i in range(1, len(comm_h)) if comm_h[i] > comm_h[i - 1] + 1e-12)
    log(f"  committed ema_held values: {len(comm_h):,}; increases over the run: {inc}"
        f"   -> P109b {'PASS' if inc == 0 else 'FAIL'}")
    if comm_h:
        log(f"  committed ema_held first {comm_h[0]:.4f} -> last {comm_h[-1]:.4f}"
            f"   (the probe loss over the same rows: {rows[0]['probe_loss']:.4f} -> {rows[-1]['probe_loss']:.4f})")
        log(f"  acceptance set is therefore 'a new low of the smoothed probe loss': accepted {len(comm_h):,}"
            f" of {n:,} steps = {len(comm_h) / n:.4f}")

    # ---------- 3. the discriminating replay: commit every step (P109d) ----------
    log("")
    log("--- 3. THE SAME LOGGED SERIES, EMA COMMITTED ON EVERY STEP (the fix under test) — P109d / P109e ---")
    log("  tol      refused fraction     refused/total        (same logged probe series, same beta)")
    for tol in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
        flags, _, _ = replay(rows, ref_policy="every_step", tol=tol, variant="ema_band", beta=a.beta)
        log(f"  {tol:<8.3f} {frac(flags):.4f}              {sum(flags):,}/{n:,}")
    log("  for reference, the SHIPPED policy (commit on accept):")
    for tol in (0.0, 0.01, 0.05):
        flags, _, _ = replay(rows, ref_policy="on_accept", tol=tol, variant="ema_band", beta=a.beta)
        log(f"  {tol:<8.3f} {frac(flags):.4f}              {sum(flags):,}/{n:,}")

    # ---------- 4. the memorisation-signature detector (P109f) ----------
    log("")
    log("--- 4. DETECTOR 3a 'gap_level' (refuse while train EMA < held EMA*(1-tol)) — P109f ---")
    log("  POLITELY STATED BEFORE READING: P109f predicted ~0.00 refusals; the logged gap changes sign, so a")
    log("  0.00 reading is not expected. The measured numbers are printed as they are.")
    for policy in ("on_accept", "every_step"):
        for tol in (0.0, 0.01):
            flags, _, _ = replay(rows, ref_policy=policy, tol=tol, variant="gap_level", beta=a.beta)
            log(f"  commit={policy:<11} tol={tol:<6.3f} refused {frac(flags):.4f}  ({sum(flags):,}/{n:,})")

    # ---------- 5. direction of the probe series (what the gate is looking at) ----------
    log("")
    log("--- 5. WHAT THE GATE IS COMPARING: the probe series itself, per epoch ---")
    for ep in epochs:
        sub = [float(r["probe_loss"]) for r in rows if int(r["epoch"]) == ep]
        if not sub:
            continue
        log(f"  epoch {ep}: probe loss {sub[0]:.4f} -> {sub[-1]:.4f} (min {min(sub):.4f}); "
            f"batch loss {float([r for r in rows if int(r['epoch']) == ep][0]['train_loss']):.4f} -> "
            f"{float([r for r in rows if int(r['epoch']) == ep][-1]['train_loss']):.4f}")
    log("")
    log("NOTHING WAS WRITTEN INTO ANY RUN. No model, no corpus, no GPU, no engine edit.")
    log("=" * 110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
