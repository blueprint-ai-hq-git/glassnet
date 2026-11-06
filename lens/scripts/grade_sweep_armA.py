#!/usr/bin/env python3
"""grade_sweep_armA.py - grade arm A of the ours-method settings sweep against the numbers IT printed.

DETERMINISTIC, TEXT-ONLY. No checkpoint is opened, no model is loaded, no GPU is touched, no training is
done. Reads:
  lens/runs/chat/ours_arm_A_ours_tol0_history.jsonl        arm A, one record per finished epoch
  lens/runs/chat/ours_arm_A_ours_tol0_gates.jsonl           arm A's per-step gap-gate log
  lens/runs/chat/ours_arm_A_ours_tol0_agreement.jsonl       arm A's per-step agreement log
  lens/runs/chat/ct_d256_full169_fix3_history.jsonl         the twin (plain loop, same corpus/seed, 3 epochs)
  lens/runs/OURS-SWEEP.txt                                  the arm's own console block (summary lines)
Appends to lens/runs/GRADE-OURS-ARM-A.txt. Predictions: lens/runs/SWEEP-GRADE-PREDICTIONS-P91.txt (written
BEFORE this grader first ran). Re-runnable; the arm may still be training, so only finished epochs are graded.

WHY PER-EPOCH COUNTS AND NOT A WHOLE-FILE RECOUNT (found while writing this, 2026-10-24): the per-step logs
APPEND ACROSS EPOCHS, so a whole-file recount after epoch 2 starts is a number nobody can use. P91a said
"recount the logs"; the correct recount is per epoch, using the record's own "epoch" field. Recorded, not
smoothed.
"""
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path("/run/media/yahya/HARD DISK 4/GlassNet")
RUNS = ROOT / "lens/runs"
CHAT = RUNS / "chat"
ARM_HIST = CHAT / "ours_arm_A_ours_tol0_history.jsonl"
ARM_GATES = CHAT / "ours_arm_A_ours_tol0_gates.jsonl"
ARM_AGREE = CHAT / "ours_arm_A_ours_tol0_agreement.jsonl"
TWIN_HIST = CHAT / "ct_d256_full169_fix3_history.jsonl"
SWEEP_CONSOLE = RUNS / "OURS-SWEEP.txt"
OUT = RUNS / "GRADE-OURS-ARM-A.txt"

# arm A's own recipe, from lens/scripts/run_ours_arm.sh (read, not remembered)
ARM_WINDOWS = 2_000_000
ARM_BATCH = 256
TWIN_WINDOWS = 16_878_132  # the full-corpus twin ftrained on the same corpus, printed in its own runner


def jload(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def count_log(path):
    """per-epoch checked/refused/accepted from a per-step jsonl"""
    per = {}
    for d in jload(path):
        e = d.get("epoch")
        slot = per.setdefault(e, {"checked": 0, "refused": 0, "accepted": 0, "max_step": 0})
        slot["checked"] += 1
        if d.get("refused") is True:
            slot["refused"] += 1
        elif d.get("accepted") is True:
            slot["accepted"] += 1
        slot["max_step"] = max(slot["max_step"], d.get("step") or 0)
    return per


def console_summaries():
    """the trainer's own 'gate:' / 'agreement:' summary lines from the arm-A block of OURS-SWEEP.txt"""
    text = SWEEP_CONSOLE.read_text(errors="replace").splitlines()
    gate, agree = [], []
    in_arm_a = False
    for ln in text:
        if ln.startswith("ARM "):
            in_arm_a = ln.startswith("ARM A_ours_tol0")
            continue
        if not in_arm_a:
            continue
        m = re.match(r"\s*gate:\s*checked ([\d,]+)\s*refused ([\d,]+)\s*\(([\d.]+)%\)\s*accepted ([\d,]+)", ln)
        if m:
            gate.append([int(x.replace(",", "")) for x in m.groups()[:2]] + [float(m.group(3))] + [int(m.group(4).replace(",", ""))])
            continue
        m = re.match(r"\s*agreement:\s*checked ([\d,]+)\s*refused ([\d,]+)\s*\(([\d.]+)%\)\s*accepted ([\d,]+)", ln)
        if m:
            agree.append([int(x.replace(",", "")) for x in m.groups()[:2]] + [float(m.group(3))] + [int(m.group(4).replace(",", ""))])
    return gate, agree


def line(tag, ok, text):
    print(f"  [{tag}] {text}")


def main():
    out = []
    def w(s=""):
        out.append(s)

    arm = jload(ARM_HIST)
    twin = jload(TWIN_HIST)
    gper = count_log(ARM_GATES)
    aper = count_log(ARM_AGREE)
    gsum, asum = console_summaries()

    w("=" * 110)
    w("GRADE OF ARM A OF THE OURS-METHOD SETTINGS SWEEP   (grade_sweep_armA.py, text-only, no GPU)")
    w(f"  arm A : {ARM_HIST}   ({len(arm)} finished epoch record(s))")
    w(f"  twin  : {TWIN_HIST}   ({len(twin)} finished epoch record(s))")
    w("=" * 110)

    w("")
    w("1. THE ARM'S OWN EPOCH LINES (read from its history file, not from memory)")
    for r in arm:
        w(f"   ep {r['epoch']}  train acc {r['train_acc']*100:.2f}%   ASST(by-position) {r['val_asst_acc']*100:.4f}% "
          f"(n={r['val_asst_n']})   LEGACY by-id {r['val_masked_acc']*100:.4f}% (n={r['val_masked_n']})   "
          f"VAL ppl {r['val_ppl']:.2f}   opt {r['opt']} lr {r['lr']}   {r['seconds']:.1f} s")

    w("")
    w("2. PER-EPOCH RECOUNT OF THE PER-STEP LOGS  (P91a - per epoch, because the logs append across epochs)")
    finished = {r["epoch"] for r in arm}
    for e in sorted(gper):
        g, a = gper[e], aper.get(e, {"checked": 0, "refused": 0, "accepted": 0, "max_step": 0})
        if e not in finished:
            w(f"   epoch {e}: ** IN PROGRESS - no finish record in the history file yet, so the counts below are")
            w(f"             PARTIAL (the arm is still training). They are printed for surveillance only and are")
            w(f"             NOT compared with any printed summary.")
        w(f"   epoch {e}: gap gate checked {g['checked']:,} refused {g['refused']:,} accepted {g['accepted']:,} "
          f"(ids up to step {g['max_step']:,})")
        w(f"            agreement checked {a['checked']:,} refused {a['refused']:,} accepted {a['accepted']:,} "
          f"(ids up to step {a['max_step']:,})")
        ident_g = g["checked"] == g["refused"] + g["accepted"]
        ident_a = a["checked"] == a["refused"] + a["accepted"]
        w(f"            identity checked = refused + accepted: gap {ident_g}, agreement {ident_a}")
        if e - 1 < len(gsum) and e - 1 < len(asum):
            gc, gr, gpct, ga = gsum[e - 1]
            ac, ar, apct, aa = asum[e - 1]
            w(f"            the trainer's OWN printed summary for this epoch: gate checked {gc:,} refused {gr:,} "
              f"({gpct}%) accepted {ga:,} | agreement checked {ac:,} refused {ar:,} ({apct}%) accepted {aa:,}")
            ok = (gc, gr, ga) == (g["checked"], g["refused"], g["accepted"]) and (ac, ar, aa) == (a["checked"], a["refused"], a["accepted"])
            w(f"            REPRODUCTION: {'MATCH - the stored counts are the raw counts' if ok else 'MISMATCH'}")
            wer = f"{g['refused'] / g['checked'] * 100:.1f}%" if g["checked"] else "n/a"
            w(f"            -> weight updates actually applied in epoch {e}: {g['accepted']:,} of "
              f"{a['checked']:,} steps = {g['accepted'] / a['checked'] * 100:.1f}%   (gap gate refused {wer})")
        else:
            w("            (no printed summary line found for this epoch in OURS-SWEEP.txt)")

    w("")
    w("3. AGAINST THE TWIN AT MATCHED EPOCH  (P91b, P91c)")
    for r in arm:
        e = r["epoch"]
        t = next((x for x in twin if x["epoch"] == e), None)
        if t is None:
            continue
        d_asst = (r["val_asst_acc"] - t["val_asst_acc"]) * 100
        w(f"   ep {e}:  arm A by-position {r['val_asst_acc']*100:.4f}%  vs  twin {t['val_asst_acc']*100:.4f}%"
          f"   -> {d_asst:+.2f} points      [arm hist / twin hist]")
        w(f"           arm A train {r['train_acc']*100:.2f}%  vs  twin train {t['train_acc']*100:.2f}%")
    if arm and twin:
        a_ws = ARM_WINDOWS / arm[0]["seconds"]
        t_ws = TWIN_WINDOWS / twin[0]["seconds"]
        w("")
        w(f"   COST, same metric both sides (windows/s at 2 cores, epoch 1): arm A {a_ws:.1f} windows/s "
          f"({ARM_WINDOWS:,} windows / {arm[0]['seconds']:.1f} s)  against the twin {t_ws:.1f} windows/s "
          f"({TWIN_WINDOWS:,} windows / {twin[0]['seconds']:.1f} s)  -> {t_ws / a_ws:.1f}x per window")

    w("")
    w("4. VERDICTS  (numbers printed beside each claim; a FAIL is called a FAIL)")
    if arm:
        r = arm[0]
        t = next((x for x in twin if x["epoch"] == 1), None)
        gap = (r["val_asst_acc"] - t["val_asst_acc"]) * 100 if t else float("nan")
        w(f"   P91a (CONTROL, per epoch) : {'PASS' if all((gc, gr, ga) == (gper[e]['checked'], gper[e]['refused'], gper[e]['accepted']) for e, (gc, gr, gp, ga) in enumerate(gsum, start=1) if e in gper) else 'FAIL'}")
        w(f"         (worded in P91 as a whole-file recount: that wording is WRONG because the file appends across")
        w(f"          epochs - the per-epoch split above is the check that can pass or fail. Recorded, not smoothed.)")
        w(f"   P91b (reproduction)       : arm A ep1 by-position {r['val_asst_acc']*100:.4f}% / twin {t['val_asst_acc']*100:.4f}% -> {'PASS' if abs(round(r['val_asst_acc']*100,4) - 12.2888) < 0.001 and abs(round(t['val_asst_acc']*100,4) - 32.5222) < 0.001 else 'FAIL'}")
        ws = TWIN_WINDOWS / twin[0]['seconds'] / (ARM_WINDOWS / r['seconds']) if r['seconds'] else float('nan')
        w(f"   P91c (cost band 30-50x)   : measured {ws:.1f}x -> {'PASS' if 30 <= ws <= 50 else 'FAIL'}")
        g1 = gper.get(1)
        rate = g1['refused'] / g1['checked'] * 100 if g1 and g1['checked'] else float('nan')
        w(f"   P91d (gap refusal > 70%)  : measured {rate:.1f}% -> {'PASS' if rate > 70 else 'FAIL'}")
        w("")
        w(f"   NOT CLAIMED: this grades ONE setting (--gate-tol 0.0) at 2M windows. It does not decide whether")
        w(f"   loosening the tolerance recovers the accuracy; that is the re-queued arms B and C.")
    else:
        w("   no epoch record yet - nothing to grade")

    text = "\n".join(out) + "\n"
    with open(OUT, "a") as f:
        f.write(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
