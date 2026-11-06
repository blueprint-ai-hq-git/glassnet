#!/usr/bin/env python3
"""H2L (the best d=256 recipe with --epochs 6, concept supervision OFF) GRADED FROM ITS OWN FILES.

TEXT ONLY: no checkpoint is opened, no model is built, no GPU is touched. Every number is printed BESIDE the file
it was read from. Predictions: lens/runs/H2L-PREDICTIONS-P104.txt (written 15:4x, BEFORE the runner file existed).

The three references this arm is measured against, all read from disk here (never hardcoded except where the
prediction file itself quoted them):
  H2      lens/runs/chat/h2_noconcept_169_3_history.jsonl      (3 epochs, concept OFF  — the arm this extends)
  H5L     lens/runs/chat/ct_d256_full169_6ep_history.jsonl     (6 epochs, concept ON   — the paired arm, P104e)
  TWIN    lens/runs/chat/ct_d256_full169_fix3_history.jsonl    (3 epochs, concept ON   — the stored reference)

Verdicts this file can carry, and nothing else:
  P104a  config control: 5,908,686 params, concept 0.0000 in every row, val_asst_n = 58,191 in every row.
  P104b  ep1 is the STRICT reproduction control against H2's own ep1 33.0000%, read against the drift floor
         measured this night (lens/runs/DRIFT-NUMBERS.txt), NOT against the 0.01 points P102c wrongly promised.
         ep2/ep3 are lr-CONFOUNDED (this run anneals over 6 epochs) and are PRINTED WITH NO VERDICT.
  P104c  ep6 by position, predicted 36.0-38.0%; FALSIFIER at or below H2's 35.7220%.
  P104d  memorisation: gap under +6 points at ep6, no peak-then-decline before ep6.
  P104e  paired against H5L's ep6 (concept ON, same 6-epoch schedule): predicted AT OR ABOVE - 1.0 point.
  P104f  cost: six epochs at H2's own measured rate = 0.95-1.25 h.
  P104g  talk: 0/6 declared in advance; mean pad-free <unk> predicted within 2 points of H2's 0.7%; step-8
         free-running predicted within 1 point of the twin's 3.9%. The by-eye count is a HUMAN field: this
         script prints what it can read and never invents the eye count.
Anything not on disk prints UNMEASURED with the file it will come from: a missing file can never become a pass.
"""
import json
import os
from datetime import datetime

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
CHAT = os.path.join(ROOT, "lens/runs/chat")
OUT = os.path.join(ROOT, "lens/runs/H2L-NUMBERS.txt")
CONSOLE = os.path.join(ROOT, "lens/runs/H2L-D256-6EP.txt")
PADFREE = os.path.join(ROOT, "lens/runs/PADFREE-TALK-H2L.txt")
TALKFAIR = os.path.join(ROOT, "lens/runs/TALK-FAIR-H2L.txt")
H2_PADFREE = os.path.join(ROOT, "lens/runs/PADFREE-TALK-H2.txt")
DRIFT = os.path.join(ROOT, "lens/runs/DRIFT-NUMBERS.txt")
ARM = os.path.join(CHAT, "h2l_noconcept_169_6ep_history.jsonl")
H2 = os.path.join(CHAT, "h2_noconcept_169_3_history.jsonl")
H5L = os.path.join(CHAT, "ct_d256_full169_6ep_history.jsonl")
TWIN = os.path.join(CHAT, "ct_d256_full169_fix3_history.jsonl")
FLOOR = 0.0739          # points by position, measured: lens/runs/DRIFT-NUMBERS.txt
H2_EP1 = 33.0000        # % by position, quoted in the P104b prediction from h2_noconcept_169_3_history.jsonl
H2_EP3 = 35.7220        # % by position, the P104c falsifier
H5L_EP6 = 36.6878       # % by position, the P104e comparison (ct_d256_full169_6ep_history.jsonl)
H2_UNK = 0.7            # % mean pad-free <unk>, lens/runs/PADFREE-TALK-H2.txt (R2H_UNK env can override)
TWIN_STEP8 = 3.9        # % free-running step 8, lens/runs/TALK-FAIR-H2-TWIN.txt

lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def load(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def pct(v):
    return "n/a" if v is None else "%.4f%%" % (100.0 * float(v))


def main():
    say("=" * 104)
    say("H2L (concept OFF, --epochs 6) GRADED FROM ITS OWN FILES — %s" % datetime.now().strftime("%F %H:%M:%S"))
    say("grader: lens/scripts/h2l_numbers.py (TEXT ONLY: no checkpoint opened, no model, no GPU)")
    say("predictions: lens/runs/H2L-PREDICTIONS-P104.txt (written before the runner existed)")
    say("records: %s" % ARM)
    say("         %s  (H2, the 3-epoch concept-OFF arm = this arm's own reference)" % H2)
    say("         %s  (H5L, 6 epochs concept ON = the paired arm, P104e)" % H5L)
    say("         %s  (the stored 3-epoch twin)" % TWIN)
    say("drift floor for every control here: %+.4f points by position [%s]" % (FLOOR, DRIFT))
    say("=" * 104)

    arm = load(ARM)
    h2 = load(H2)
    h5l = load(H5L)
    twin = load(TWIN)
    if not arm:
        say("")
        say("EMPTY OR MISSING READ of %s — NOTHING IS GRADED." % ARM)
        with open(OUT, "a") as fh:
            fh.write("\n".join(lines) + "\n")
        return 1

    say("")
    say("--- THE ARM'S OWN ROWS, WITH H2 (its 3-epoch parent) BESIDE IT ---")
    say("%-3s %-10s %-11s %-10s %-9s %-8s %-8s %-8s %-7s" %
        ("ep", "train", "by-pos", "asst ppl", "concept", "lr", "h2 lr", "secs", "n"))
    for r in arm:
        ep = int(r["epoch"])
        p = h2[ep - 1] if h2 and ep - 1 < len(h2) else None
        say("%-3d %-10s %-11s %-10s %-9.4f %-8.4f %-8s %-8.1f %-7d" %
            (ep, pct(r.get("train_acc")), pct(r.get("val_asst_acc")), "%.3f" % r.get("val_asst_ppl", float("nan")),
             float(r.get("concept_loss", float("nan"))), float(r.get("lr", float("nan"))),
             ("%.4f" % float(p["lr"])) if p else "n/a", float(r.get("seconds", 0.0)),
             int(r.get("val_asst_n", -1))))
        if p is not None:
            say("    H2 ep%d: train %s  by-pos %s  asst ppl %.3f  concept %.4f   ->  by-pos delta %+.4f points"
                % (ep, pct(p.get("train_acc")), pct(p.get("val_asst_acc")), p.get("val_asst_ppl", float("nan")),
                   p.get("concept_loss", float("nan")),
                   (100.0 * (float(r["val_asst_acc"]) - float(p["val_asst_acc"])))))

    say("")
    say("P104a CONFIG CONTROL:")
    txt = open(CONSOLE, errors="replace").read() if os.path.exists(CONSOLE) else ""
    pline = [l.strip() for l in txt.splitlines() if "params" in l and "model:" in l]
    par = pline[0] if pline else "UNREAD (%s)" % CONSOLE
    concept_zero = all(abs(float(r.get("concept_loss", -1.0))) < 1e-12 for r in arm)
    ns = sorted({int(r.get("val_asst_n", -1)) for r in arm})
    say("  console says: %s" % par)
    say("  concept_loss 0.0000 in every row: %s   |   val_asst_n in every row: %s  -> %s"
        % (concept_zero, ns, "PASS" if (concept_zero and ns == [58191]) else "FAIL"))

    say("")
    say("P104b AT EP1 (the STRICT control: same seed, corpus, batch, window order, lr 0.4):")
    r1 = arm[0]
    d1 = 100.0 * float(r1["val_asst_acc"]) - H2_EP1
    say("  arm  by-pos %s (lr %.4f)   train_loss %.6f   concept %.6f" %
        (pct(r1["val_asst_acc"]), float(r1["lr"]), r1["train_loss"], r1["concept_loss"]))
    say("  H2   by-pos %.4f%% (lr %.4f) — %s" % (H2_EP1, 0.4, H2))
    say("  delta %+.4f points against the drift floor %+.4f -> %s"
        % (d1, FLOOR, "PASS (inside the floor)" if abs(d1) <= FLOOR else "FAIL (outside the floor)"))
    say("  NOTE ON P104b's OWN WORDING: the prediction file asks for reproduction 'to within 0.01 points' in its")
    say("  first clause and then, one line later, says the tolerance must be the measured floor. 0.01 points is")
    say("  BELOW what this machine can do (drift floor %+.4f, lens/runs/DRIFT-NUMBERS.txt), so the floor is the"
        % FLOOR)
    say("  binding number and 0.01 points is recorded as a defect in the prediction file, not as a failed control.")
    say("")
    say("P104b AT EP2/EP3 (lr-confounded — NO VERDICT, printed with both lr values):")
    for ep in (2, 3):
        if len(arm) < ep or not h2 or len(h2) < ep:
            say("  ep%d: UNMEASURED" % ep)
            continue
        r, p = arm[ep - 1], h2[ep - 1]
        say("  ep%d: arm %s (lr %.4f) against H2's %s (lr %.4f) = %+.4f points — printed, not graded: the anneal"
            % (ep, pct(r["val_asst_acc"]), float(r["lr"]), pct(p["val_asst_acc"]), float(p["lr"]),
               100.0 * (float(r["val_asst_acc"]) - float(p["val_asst_acc"]))))

    vals = [100.0 * float(r["val_asst_acc"]) for r in arm]
    say("")
    say("THE 6-EPOCH CURVE OF THIS ARM: " + " -> ".join("%.4f" % v for v in vals) + "   (%d of 6 rows)" % len(arm))
    best = max(range(len(vals)), key=lambda i: vals[i])
    say("  best so far: ep%d at %.4f%%; still improving at the last row written: %s"
        % (best + 1, vals[best], "YES" if best == len(arm) - 1 else "NO"))

    say("")
    say("P104c (ep6 by position predicted 36.0-38.0%%; FALSIFIER at or below H2's %.4f%%):" % H2_EP3)
    if len(arm) < 6:
        say("  UNMEASURED — %d of 6 rows written (%s)." % (len(arm), ARM))
    else:
        v6 = vals[5]
        say("  ep6 %.4f%%  |  band 36.0-38.0  -> %s  |  falsifier (<= %.4f%%): %s  |  over H2's ep3 %+.4f"
            % (v6, "INSIDE" if 36.0 <= v6 <= 38.0 else "OUTSIDE", H2_EP3,
               "FIRED" if v6 <= H2_EP3 else "did not fire", v6 - H2_EP3))

    say("")
    say("P104d (memorisation: gap under +6 points at ep6, no peak-then-decline before ep6):")
    gaps = [100.0 * (float(r["train_acc"]) - float(r["val_asst_acc"])) for r in arm]
    say("  gaps ep1..ep%d: " % len(gaps) + " / ".join("%+.3f" % g for g in gaps))
    if len(arm) < 6:
        say("  UNMEASURED at ep6 — %d of 6 rows." % len(arm))
    else:
        say("  gap at ep6 %+.3f against the +6.0 bar -> %s" % (gaps[5], "HELD" if gaps[5] < 6.0 else "FAIL"))
        say("  best held-out at ep%d of 6 -> %s" % (best + 1,
                                                    "NO PEAK-THEN-DECLINE (best is the last row)"
                                                    if best == 5 else "A PEAK AT ep%d" % (best + 1)))

    say("")
    say("P104e (paired against H5L, concept ON, same 6-epoch schedule; predicted AT OR ABOVE - 1.0 point):")
    if len(arm) < 6 or not h5l or len(h5l) < 6:
        say("  UNMEASURED — needs ep6 on both arms.")
    else:
        v = vals[5]
        say("  H2L ep6 %s (concept OFF)  vs  H5L ep6 %.4f%% (concept ON)  ->  %+.4f points  ->  %s"
            % (pct(arm[5]["val_asst_acc"]), H5L_EP6, v - H5L_EP6,
               "PASS" if v >= H5L_EP6 - 1.0 else "FAIL (concept-off is more than 1.0 pt worse at 6 epochs)"))

    say("")
    say("P104f (cost: predicted 0.95-1.25 h for six epochs at H2's own measured rate):")
    secs = [float(r.get("seconds", 0.0)) for r in arm]
    say("  per-epoch seconds: " + " / ".join("%.1f" % s for s in secs) + "   total %.1f s = %.3f h"
        % (sum(secs), sum(secs) / 3600.0))
    if len(secs) == 6:
        h = sum(secs) / 3600.0
        say("  -> %s the 0.95-1.25 h band" % ("INSIDE" if 0.95 <= h <= 1.25 else "OUTSIDE"))

    say("")
    say("P104g (talk — THE decider; 0/6 declared in advance; <unk> within 2 pts of H2's %.1f%%; step-8 within" % H2_UNK)
    say("      1 pt of the twin's %.1f%%):" % TWIN_STEP8)
    if not os.path.exists(PADFREE):
        say("  UNMEASURED — %s does not exist yet (the chained pad-free instrument writes it at the end of the job)."
            % PADFREE)
    else:
        t = open(PADFREE, errors="replace").read()
        say("  file: %s (%d B)" % (PADFREE, len(t)))
        for key in ("SHIP SUMMARY", "mean <unk>"):
            for h in [l.strip() for l in t.splitlines() if key in l][:8]:
                say("  %s" % h)
    say("  BY EYE (a human field, filled in by the wake that reads the six printed blocks — never invented here):")
    say("    %s" % (os.environ.get("H2L_EYE") or
                    "UNFILLED BY THIS SCRIPT"))
    say("  the twin control block this arm is read against: %s" % os.path.join(ROOT, "lens/runs/TALK-FAIR-H2L.txt"))
    say("  file %s: %s" % (TALKFAIR, "present" if os.path.exists(TALKFAIR) else "NOT ON DISK YET"))
    if os.path.exists(TALKFAIR):
        for h in [l.strip() for l in open(TALKFAIR, errors="replace").read().splitlines()
                  if "step" in l.lower() and "%" in l][:10]:
            say("    %s" % h)

    say("")
    say("NOTHING SHIPS FROM THIS ARM unless the by-eye count answers real prompts; a 0/6 is NOT talk progress,")
    say("whatever the accuracy column says (STATE.md rule, and P104g declares it in advance).")
    say("=" * 104)
    with open(OUT, "a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("appended to %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
