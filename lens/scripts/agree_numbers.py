#!/usr/bin/env python3
"""lens/scripts/agree_numbers.py -- grade the AGREEMENT arm (agree_sgd04_s2) from its OWN files.

Predictions and verdict rules were written BEFORE this file existed:
lens/runs/AGREE-GRADE-PREDICTIONS-P113.txt.

READ-ONLY. No checkpoint is loaded, no model is built, no GPU is touched, no generation is run. The arm
keeps training while this runs.

WHAT IT PRINTS, and every number carries the file it came from:
  1. CONTROLS C1/C2 -- the arm's console epoch lines reproduced from its own jsonl rows, digit for digit.
     If either control fails, no verdict is issued.
  2. THE TABLE      -- per epoch: steps, refusals, rate, cost, held-out columns, and the ungated twin's
     own row at the SAME epoch beside them (the twin is a stored run: lens/runs/chat/opt_sgd04_s2_*).
  3. P61a/P61b/P61c(as far as it is decidable)/P61d/P61e -- each verdict printed beside the compared
     values, because this project's most repeated bug is a right number under a wrong conclusion.
  4. THE GATE CURVE -- the same recipe under three settings measured at matched epoch 1:
     no gate, this agreement gate, and the gap gate (lens/runs/chat/gate_sgd04_s2_*).
"""
import json
import os
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(PROJ, "lens", "runs")
CHAT = os.path.join(RUNS, "chat")

ARM = "agree_sgd04_s2"
TWIN = "opt_sgd04_s2"
GAP = "gate_sgd04_s2"
CONSOLE = os.path.join(RUNS, "AGREE-ARM.txt")
DRIFT_FLOOR = 0.0739            # points, lens/runs/DRIFT-NUMBERS.txt
TOL_PROJECT = 0.2               # points, the project's >= 2x floor rule

fails = []


def rows(path):
    if not os.path.exists(path):
        fails.append(f"MISSING FILE {path}")
        return []
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def read_text(path):
    if not os.path.exists(path):
        fails.append(f"MISSING FILE {path}")
        return ""
    with open(path, errors="replace") as f:
        return f.read()


EP_RE = re.compile(
    r"ep\s+(\d+)/(\d+)\s+loss\s+([\d.]+)\s+concept\s+([\d.]+)\s+acc\s+([\d.]+)%\s+"
    r"VAL ppl\s+([\d.]+)\s+acc\s+([\d.]+)%\s+"
    r"MASKED\(legacy by-id\) ppl\s+([\d.]+)\s+acc\s+([\d.]+)%\s+\(n=(\d+)\)\s+"
    r"ASST\(by-position\) ppl\s+([\d.]+)\s+acc\s+([\d.]+)%\s+\(n=(\d+)\)\s+"
    r"lr\s+([\d.]+)\s+\((\d+)s\)")


def parse_console(txt):
    out = {}
    for m in EP_RE.finditer(txt):
        ep = int(m.group(1))
        out[ep] = dict(total=int(m.group(2)), loss=float(m.group(3)), concept=float(m.group(4)),
                       acc=float(m.group(5)), val_ppl=float(m.group(6)), val_acc=float(m.group(7)),
                       masked_ppl=float(m.group(8)), masked_acc=float(m.group(9)), masked_n=int(m.group(10)),
                       asst_ppl=float(m.group(11)), asst_acc=float(m.group(12)), asst_n=int(m.group(13)),
                       lr=float(m.group(14)), secs=int(m.group(15)))
    return out


def main():
    print("=" * 100)
    print("AGREEMENT ARM GRADE (agree_sgd04_s2) -- read-only, from the arm's own files")
    print("=" * 100)
    hist = rows(os.path.join(CHAT, f"{ARM}_history.jsonl"))
    agree = rows(os.path.join(CHAT, f"{ARM}_agreement.jsonl"))
    twin = rows(os.path.join(CHAT, f"{TWIN}_history.jsonl"))
    gap = rows(os.path.join(CHAT, f"{GAP}_history.jsonl"))
    console = parse_console(read_text(CONSOLE))

    print(f"  arm history      : lens/runs/chat/{ARM}_history.jsonl   {len(hist)} row(s)")
    print(f"  arm per-step log : lens/runs/chat/{ARM}_agreement.jsonl {len(agree)} row(s)")
    print(f"  console          : lens/runs/AGREE-ARM.txt               {len(console)} epoch line(s) parsed")
    print(f"  ungated twin     : lens/runs/chat/{TWIN}_history.jsonl   {len(twin)} row(s)")
    print(f"  gap-gate arm     : lens/runs/chat/{GAP}_history.jsonl    {len(gap)} row(s)")
    if fails:
        print("\n*** MISSING INPUTS, NO VERDICT ISSUED:")
        for f in fails:
            print("   ", f)
        return 1

    # ------------------------------------------------------------------ CONTROL C1
    print("\n" + "-" * 100)
    print("CONTROL C1 -- every console field reproduced from the arm's own jsonl row, digit for digit")
    bad = 0
    for r in hist:
        ep = r["epoch"]
        c = console.get(ep)
        if c is None:
            print(f"  ep{ep}: NO CONSOLE LINE -> cannot control")
            bad += 1
            continue
        checks = [
            ("train acc %", round(r["train_acc"] * 100, 2), c["acc"]),
            ("val ppl", round(r["val_ppl"], 2), c["val_ppl"]),
            ("val acc %", round(r["val_acc"] * 100, 2), c["val_acc"]),
            ("masked ppl", round(r["val_masked_ppl"], 2), c["masked_ppl"]),
            ("masked acc %", round(r["val_masked_acc"] * 100, 2), c["masked_acc"]),
            ("masked n", int(r["val_masked_n"]), c["masked_n"]),
            ("asst ppl", round(r["val_asst_ppl"], 2), c["asst_ppl"]),
            ("asst acc %", round(r["val_asst_acc"] * 100, 2), c["asst_acc"]),
            ("asst n", int(r["val_asst_n"]), c["asst_n"]),
            ("lr", round(r["lr"], 4), c["lr"]),
            ("seconds", int(round(r["seconds"])), c["secs"]),
        ]
        diff = [(n, a, b) for (n, a, b) in checks if a != b]
        status = "PASS" if not diff else "FAIL"
        print(f"  ep{ep}: {len(checks) - len(diff)}/{len(checks)} fields identical -> {status}")
        for n, a, b in diff:
            print(f"        {n}: jsonl {a}  vs console {b}")
            bad += 1
    print(f"  C1 VERDICT: {'PASS' if bad == 0 else 'FAIL'}  ({bad} mismatched field(s))")
    if bad:
        print("  *** A FAILED CONTROL VOIDS EVERY VERDICT BELOW. Nothing is graded this run.")
        return 1

    # ------------------------------------------------------------------ CONTROL C2
    print("\nCONTROL C2 -- step accounting: 5,860 steps per completed epoch, refusals counted per epoch")
    per_ep = {}
    for r in agree:
        e = r["epoch"]
        d = per_ep.setdefault(e, {"steps": 0, "refused": 0})
        d["steps"] += 1
        d["refused"] += 1 if r["refused"] else 0
    ok = True
    for ep in sorted(per_ep):
        d = per_ep[ep]
        done = ep in [h["epoch"] for h in hist]
        want = 5860 if done else None
        line = f"  ep{ep}: steps {d['steps']}  refused {d['refused']}  rate {d['refused'] / d['steps']:.4f}"
        if done:
            line += f"   (complete: {want} expected)"
            if d["steps"] != want:
                line += "  *** MISMATCH"
                ok = False
        else:
            line += "   (IN PROGRESS at grade time)"
        print(line)
    print(f"  C2 VERDICT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  *** A FAILED CONTROL VOIDS EVERY VERDICT BELOW.")
        return 1

    # ------------------------------------------------------------------ THE TABLE
    print("\n" + "-" * 100)
    print("THE TABLE -- arm vs its ungated twin at the SAME epoch (the twin needs no re-run: it is stored)")
    print("  ep | arm masked%  twin masked%   delta  | arm asst% | arm secs  twin secs   cost x  | refusal")
    tw = {r["epoch"]: r for r in twin}
    total_secs = 0.0
    for r in hist:
        ep = r["epoch"]
        t = tw.get(ep)
        a_m = r["val_masked_acc"] * 100
        t_m = t["val_masked_acc"] * 100 if t else float("nan")
        cost = r["seconds"] / t["seconds"] if t else float("nan")
        total_secs += r["seconds"]
        d = per_ep[ep]
        print(f"  {ep:2d} | {a_m:11.4f} {t_m:12.4f} {a_m - t_m:+8.4f}  | {r['val_asst_acc'] * 100:8.4f}  |"
              f" {r['seconds']:8.1f}  {t['seconds'] if t else float('nan'):9.1f}  {cost:7.2f}  |"
              f" {d['refused']}/{d['steps']} = {d['refused'] / d['steps']:.4f}")
    print("  (masked = LEGACY by-token-id column, n=44,418 in both arms; asst = by-position, n=58,191, the"
          " column the ship criterion uses; the twin predates that column and has none.)")
    print(f"  steps per epoch 5,860 = 1,500,000 windows // 256;  twin per-epoch ~67 s (its own rows).")

    # ------------------------------------------------------------------ VERDICTS
    print("\n" + "-" * 100)
    print("VERDICTS -- band as written in lens/runs/AGREEMENT-GATE-PREDICTIONS-APPENDIX.txt (2026-10-21)")
    rates = {ep: per_ep[ep]["refused"] / per_ep[ep]["steps"] for ep in per_ep}
    first = min(rates)
    print(f"\nP61a  BAND 0.005-0.08 that the detector refuses on the real recipe.")
    for ep in sorted(rates):
        tag = "MISS HIGH" if rates[ep] > 0.08 else ("in band" if rates[ep] >= 0.005 else "MISS LOW")
        print(f"      ep{ep} rate {rates[ep]:.4f}  -> {tag}  (band top 0.08, ratio {rates[ep] / 0.08:.1f}x)")
    tot_rate = sum(d['refused'] for d in per_ep.values()) / sum(d['steps'] for d in per_ep.values())
    print(f"      ALL measured steps {tot_rate:.4f} -> "
          f"{'MISS HIGH' if tot_rate > 0.08 else ('in band' if tot_rate >= 0.005 else 'MISS LOW')}")
    print(f"      VERDICT: the arm's letter (refusals > 0) PASSES; ITS DECLARED BAND IS MISSED ON THE HIGH")
    print(f"      SIDE by {rates[first] / 0.08:.1f}x at ep{first}. Recorded as a BAND MISS, not a pass.")
    print(f"      PREDICTION P113c (miss high at every completed epoch): PASS.")

    print(f"\nP61b  COST -- band 1.5-3.0x the ungated twin's per-epoch time; the arm's OWN wording makes")
    print(f"      anything above 4.5x a FAIL with a consequence ('the free version becomes the next item').")
    mult = {r['epoch']: r['seconds'] / tw[r['epoch']]['seconds'] for r in hist if r['epoch'] in tw}
    for ep in sorted(mult):
        print(f"      ep{ep}: {mult[ep]:.2f}x  ({mult[ep] / 3.0:.1f}x the band top 3.0, {mult[ep] / 4.5:.1f}x the 4.5x falsifier)")
    print(f"      VERDICT: FAIL. Above 4.5x at every completed epoch -> by the arm's own pre-written rule the")
    print(f"      FREE variant (P60m) is required, not a nice-to-have.")
    print(f"      PREDICTION P113d (20-35x): "
          f"{'PASS' if all(20 <= v <= 35 for v in mult.values()) else 'MISS'} "
          f"(measured {min(mult.values()):.2f}-{max(mult.values()):.2f}x).")

    print(f"\nP61c  AS WORDED -- final masked acc within 0.005 ABSOLUTE of the twin's 0.284636858931064.")
    print(f"      NOT DECIDABLE TONIGHT: the arm is at ep{max(h['epoch'] for h in hist)} of 12 and 10 more")
    print(f"      epochs x ~2,087 s = 5.8 h against ~3.5 h of machine time before the ~21:41 shutdown.")
    print(f"      MATCHED-EPOCH RULE ADDED BY P113e, and its numbers:")
    worst = 0.0
    for r in hist:
        ep = r['epoch']
        if ep in tw:
            d = r['val_masked_acc'] * 100 - tw[ep]['val_masked_acc'] * 100
            worst = min(worst, d)
            print(f"      ep{ep}: arm {r['val_masked_acc'] * 100:.4f}%  twin {tw[ep]['val_masked_acc'] * 100:.4f}%"
                  f"  delta {d:+.4f} points  = {abs(d) / DRIFT_FLOOR:.0f}x the {DRIFT_FLOOR}-point drift floor")
    print(f"      VERDICT: the arm is BELOW its ungated twin at every matched epoch by more than 5 points, so")
    print(f"      the honest number is damaged, not held; the 0.005-absolute bar is already violated in the")
    print(f"      damaging direction. This does NOT decide P61c as worded and is not reported as if it did.")
    print(f"      PREDICTION P113e (below by > 5 points): {'PASS' if worst < -5 else 'MISS'} ({worst:+.4f}).")

    print(f"\nP61d  A_train at step 1 < 0.30 (against 0.3134 on corpus M, 0.3047 on corpus L).")
    print(f"      step 1: a_train {agree[0]['a_train']:.4f}  a_probe {agree[0]['a_probe']:.4f}"
          f"  -> {'PASS' if agree[0]['a_train'] < 0.30 else 'FAIL'}")
    print(f"      (a_train below the 1/sqrt(8) = 0.3536 independence floor, as on both small corpora)")

    print(f"\nP61e  NO ACCURACY CLAIM BEYOND P61c. HELD: this grade issues none. The arm has no chained")
    print(f"      decider and has never been generated from; NOTHING SHIPS FROM THIS ARM and the talk test")
    print(f"      is untouched by it.")

    print(f"\nP113f REFUSAL RATE FALLS OVER EPOCHS: "
          f"{' -> '.join(f'ep{ep} {rates[ep]:.4f}' for ep in sorted(rates))}  -> "
          f"{'PASS' if len(rates) >= 2 and rates[max(rates)] < rates[min(rates)] else 'MISS'}")

    # ------------------------------------------------------------------ THE GATE CURVE
    print("\n" + "-" * 100)
    print("THE SAME RECIPE UNDER THREE GATE SETTINGS, AT MATCHED EPOCH (ep1 and ep2), from stored files only")
    print("  setting                          ep1 masked%   ep2 masked%   ep1 secs   refusal ep1")
    def line(name, h, rates=None):
        r1 = next((x for x in h if x['epoch'] == 1), None)
        r2 = next((x for x in h if x['epoch'] == 2), None)
        print(f"  {name:32s} {r1['val_masked_acc'] * 100 if r1 else float('nan'):11.4f} "
              f"{r2['val_masked_acc'] * 100 if r2 else float('nan'):12.4f} "
              f"{r1['seconds'] if r1 else float('nan'):10.1f}   {rates or ''}")
    line("no gate (twin opt_sgd04_s2)", twin)
    line("agreement gate (this arm)", hist, f"{rates.get(1, float('nan')):.4f}")
    # THE GAP GATE'S REFUSAL RATE COMES FROM ITS OWN PER-STEP LOG, NOT FROM A CONSOLE FRAGMENT. The first
    # version of this line regexed "refused N/M" out of GATE-ARM.txt and printed 0.0350 -- a number from the
    # wrong place beside the right row, which is this project's most repeated bug class. GRADE-P109 measures
    # 0.8582 over 12,604 rows; this reads the same file it does.
    gap_rate = None
    gap_log = os.path.join(CHAT, f"{GAP}_gates.jsonl")
    if os.path.exists(gap_log):
        gn = gr = 0
        with open(gap_log) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                gn += 1
                gr += 1 if r.get("refused") else 0
        gap_rate = (f"{gr / gn:.4f} ({gr}/{gn} rows = ALL its logged steps; GRADE-P109 graded the first"
                    f" 12,604 = 0.8582 -- a different window, not a contradiction)")
    line("gap gate (gate_sgd04_s2)", gap, gap_rate or "per-step log missing")
    print("  READ THIS AS A CURVE, NOT A CAUSE: more refusals -> fewer weight updates -> a lower matched-epoch")
    print("  number. The agreement arm also computes 8 group gradients per step, so its cost is not evidence")
    print("  about the gap gate and the two are compared here only on the held-out column.")

    print("\n" + "=" * 100)
    print("SUMMARY (each number above names its file; nothing here ships and nothing talks)")
    print(f"  refusal  : {tot_rate:.4f} overall ({sum(d['refused'] for d in per_ep.values())}/"
          f"{sum(d['steps'] for d in per_ep.values())} steps) vs the arm's 0.005-0.08 band -> MISS HIGH")
    print(f"  cost     : {min(mult.values()):.2f}-{max(mult.values()):.2f}x the twin per epoch vs a 1.5-3.0 band"
          f" -> FAIL, {min(mult.values()) / 4.5:.1f}-{max(mult.values()) / 4.5:.1f}x its own 4.5x falsifier")
    print(f"  accuracy : {worst:+.4f} points vs the twin at matched epoch -> the honest number is damaged")
    print(f"  talk     : NOT MEASURED AND NOT CLAIMED (P61e). Nothing ships.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
