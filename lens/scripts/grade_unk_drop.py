#!/usr/bin/env python3
"""grade_unk_drop.py — P89d/P89e for the --drop-unk arm, FROM STORED TEXT ONLY.
No checkpoint is opened, no model is built, no GPU, no training. Predictions were written before the arm was
queued: lens/runs/UNK-DROP-PREDICTIONS-P89.txt. Written 2026-10-24 by the self-wake.

It prints, FOR EVERY NUMBER IT COMPARES, the file the number came from — this project's rule, and the rule its
verdict-check bugs were all about (a right number given a wrong comparison).
"""
import json
import os
import re
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"
# ENV OVERRIDES: added so this grader can be DRY-RUN against a stored arm before the real one exists (the
# project's rule: an instrument is validated before the run it decides). Defaults are the real files.
LOG = os.environ.get("UNKDROP_LOG", os.path.join(P, "lens/runs/UNK-DROP.txt"))
ARM_HIST = os.environ.get("UNKDROP_ARM_HIST", os.path.join(P, "lens/runs/chat/unk_drop_169_3_history.jsonl"))
TWIN_HIST = os.environ.get("UNKDROP_TWIN_HIST", os.path.join(P, "lens/runs/chat/ct_d256_full169_fix3_history.jsonl"))
TWIN_LOG = os.environ.get("UNKDROP_TWIN_LOG", os.path.join(P, "lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt"))
OUT = os.environ.get("UNKDROP_GRADE_OUT", os.path.join(P, "lens/runs/GRADE-P89.txt"))


class Tee:
    def __init__(self, path):
        self.f = open(path, "a")
    def w(self, s=""):
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()
    def close(self):
        self.f.close()


def rows(path):
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    T = Tee(OUT)
    T.w("=" * 94)
    T.w("GRADE OF P89d/P89e — THE --drop-unk ARM, read from its own stored log and records")
    T.w("predictions: lens/runs/UNK-DROP-PREDICTIONS-P89.txt (P89d cost, P89e held-out not damaged)")
    T.w("instrument: lens/scripts/grade_unk_drop.py (stored text only; no checkpoint, no model, no GPU)")
    T.w("=" * 94)
    T.w()

    for f in (LOG, ARM_HIST, TWIN_HIST):
        T.w(f"  {'FOUND   ' if os.path.exists(f) else 'MISSING '} {f}"
            + (f"   ({os.path.getsize(f):,} bytes)" if os.path.exists(f) else ""))
    T.w()

    arm = rows(ARM_HIST)
    twin = rows(TWIN_HIST)
    if not twin:
        T.w("TWIN RECORDS MISSING — the comparison cannot be made and NOTHING is claimed. Stop.")
        T.close()
        return
    if not arm:
        T.w("ARM RECORDS MISSING/EMPTY (the arm wrote no epoch) — nothing to grade. The arm produced no")
        T.w("artifact; that is recorded as such and no number is invented here.")
        T.close()
        return

    twin_best = max(twin, key=lambda r: r["val_asst_acc"])
    twin_asst = twin_best["val_asst_acc"] * 100
    T.w("STORED TWIN (the matched control), quoted with its file before anything is compared:")
    T.w(f"  {TWIN_HIST}")
    T.w(f"    best by BY-POSITION held-out: ep{twin_best['epoch']}  train {twin_best['train_acc']*100:.2f}%  "
        f"by-position {twin_asst:.2f}% (n={twin_best.get('val_asst_n')})  "
        f"legacy by-id {twin_best['val_masked_acc']*100:.2f}%  gap "
        f"{twin_best['train_acc']*100 - twin_asst:+.2f} points")
    T.w(f"  {TWIN_LOG} (its console log, for the epoch lines)")
    T.w()

    log_txt = open(LOG).read() if os.path.exists(LOG) else ""

    # ---------------------------------------------------------------- CONTROL
    T.w("-" * 94)
    T.w("P89 CONTROL — the arm's printed epoch lines against its own per-epoch records, and the drop-unk lines")
    T.w("-" * 94)
    printed = re.findall(r"^\s*ep\s+(\d+)/(\d+)\s+loss\s+(\S+)\s+concept\s+(\S+)\s+acc\s+([\d.]+)%\s+.*?"
                         r"ASST\(by-position\).*?acc\s+([\d.]+)%", log_txt, re.M)
    ok = True
    if not printed:
        T.w("  the log has NO epoch line yet (arm may have been stopped before its first epoch)")
        ok = False
    for ep, tot, loss, closs, tracc, asst in printed:
        r = next((x for x in arm if int(x["epoch"]) == int(ep)), None)
        if r is None:
            T.w(f"  ep{ep}: printed but NOT in the records -> MISMATCH")
            ok = False
            continue
        m1 = abs(float(tracc) - r["train_acc"] * 100) < 0.005
        m2 = abs(float(asst) - r["val_asst_acc"] * 100) < 0.005
        T.w(f"  ep{ep}/{tot}  train {tracc}% vs records {r['train_acc']*100:.2f}%  -> "
            f"{'MATCH' if m1 else 'MISMATCH'}   by-position {asst}% vs {r['val_asst_acc']*100:.2f}%  -> "
            f"{'MATCH' if m2 else 'MISMATCH'}")
        ok = ok and m1 and m2
    exits = [int(m.group(1)) for m in re.finditer(r"^(?:ARM )?EXIT=(\d+)", log_txt, re.M)]
    T.w(f"  ARM EXIT lines: {exits}  (0 = the trainer itself returned cleanly)")
    dlines = [l.strip() for l in log_txt.splitlines() if "drop-unk:" in l]
    T.w(f"  the flag's own printed lines ({len(dlines)}), verbatim, because the cut is the experiment:")
    for l in dlines:
        T.w("    " + l)
    T.w(f"  -> CONTROL {'PASS' if (ok and exits == [0]) else 'FAIL/INCOMPLETE'}")
    T.w()

    # ---------------------------------------------------------------- P89d cost
    T.w("-" * 94)
    T.w("P89d — COST: wall 1.0-1.6 h for the whole arm, checkpoint every 1800 s")
    T.w("-" * 94)
    secs = [r.get("seconds") for r in arm]
    T.w(f"  per-epoch seconds from the records ({ARM_HIST}): "
        f"{[int(s) for s in secs if s is not None]}  (sum {sum(s for s in secs if s):.1f} s)")
    walls = [int(m.group(1)) for m in re.finditer(r"wall=(\d+)s", log_txt)]
    T.w(f"  runner wall lines from {LOG}: {walls} s")
    hi = max(walls) if walls else (sum(s for s in secs if s) if secs else 0)
    cost_ok = 3600 <= hi <= 5760 and len(arm) == 3
    T.w(f"  wall used for the verdict: {hi} s = {hi/3600:.2f} h    -> "
        f"{'INSIDE' if cost_ok else 'OUTSIDE/INCOMPLETE'} the predicted 1.0-1.6 h band")
    T.w("  NOT MEASURED BY THIS INSTRUMENT: peak host RSS. It must be read from the arm's own systemd scope")
    T.w("  (systemctl --user show -p MemoryPeak glassnet-q-<stamp>.scope, the way P69b was), and is recorded")
    T.w("  as unmeasured here rather than estimated.")
    T.w()

    # ---------------------------------------------------------------- P89e held-out
    T.w("-" * 94)
    T.w(f"P89e — HELD-OUT IS NOT DAMAGED: ep3 by-position within 3.0 points of the twin's {twin_asst:.2f}%")
    T.w("-" * 94)
    T.w(f"  band = {twin_asst-3:.2f}% .. {twin_asst+3:.2f}%")
    T.w(f"  {'ep':>3} {'arm train':>10} {'arm by-pos':>11} {'arm legacy':>11} {'arm gap':>8}   "
        f"{'twin train':>10} {'twin by-pos':>11} {'twin gap':>8}")
    for r in arm:
        tw = next((x for x in twin if int(x["epoch"]) == int(r["epoch"])), None)
        a_asst = r["val_asst_acc"] * 100
        a_gap = r["train_acc"] * 100 - a_asst
        if tw:
            t_asst = tw["val_asst_acc"] * 100
            t_gap = tw["train_acc"] * 100 - t_asst
            T.w(f"  {r['epoch']:>3} {r['train_acc']*100:>9.2f}% {a_asst:>10.2f}% "
                f"{r['val_masked_acc']*100:>10.2f}% {a_gap:>+7.2f}   {tw['train_acc']*100:>9.2f}% "
                f"{t_asst:>10.2f}% {t_gap:>+7.2f}   delta by-position {a_asst - t_asst:+.2f} points")
        else:
            T.w(f"  {r['epoch']:>3} {r['train_acc']*100:>9.2f}% {a_asst:>10.2f}% "
                f"{r['val_masked_acc']*100:>10.2f}% {a_gap:>+7.2f}   (twin has no row for this epoch)")
    last = arm[-1]
    a3 = last["val_asst_acc"] * 100
    if len(arm) >= 3:
        inside = (twin_asst - 3.0) <= a3 <= (twin_asst + 3.0)
        T.w(f"  arm ep3 by-position {a3:.2f}%  against the band -> {'INSIDE (PASS)' if inside else 'OUTSIDE (FAIL)'}"
            f"   delta against the twin: {a3 - twin_asst:+.2f} points")
        T.w(f"  my P89e expectation was the LOWER half (33.5-35.5%); measured {a3:.2f}% -> "
            f"{'inside my expectation' if 33.5 <= a3 <= 35.5 else 'OUTSIDE my expectation (recorded, not rounded)'}")
        T.w(f"  my P89e companion: train accuracy BELOW the twin's ep3 {twin_best['train_acc']*100:.2f}% -> "
            f"measured {last['train_acc']*100:.2f}% -> "
            f"{'HELD' if last['train_acc'] < twin_best['train_acc'] else 'FAILED (train did not fall)'}")
    else:
        T.w(f"  THE ARM HAS {len(arm)} EPOCH ROW(S), NOT 3: P89e is graded on whatever epochs exist and the"
            f" missing ones are reported as missing, never interpolated.")
    T.w()
    T.w("THE DECIDER (P89f) IS NOT IN THIS FILE: it is the talk test, run next in this chain on the BEST")
    T.w(f"checkpoint (never the last), and graded in lens/runs/PADFREE-TALK-UNKDROP.txt and")
    T.w("lens/runs/TALK-FAIR-UNKDROP.txt by grade_unk_drop_talk.py. Nothing is shipped from this arm (P89g).")
    T.w("=" * 94)
    T.close()


if __name__ == "__main__":
    sys.exit(main())
