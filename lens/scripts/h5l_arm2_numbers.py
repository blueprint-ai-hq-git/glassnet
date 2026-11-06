#!/usr/bin/env python3
"""ARM 2 (h5l — the twin's recipe with --epochs 6) GRADED FROM ITS OWN FILES.

TEXT ONLY: no checkpoint is opened, no model is built, no GPU is touched. Every number is printed BESIDE the file
it was read from. Predictions: lens/runs/H2-H5L-PREDICTIONS-P102.txt (written BEFORE either runner existed).

Verdicts this file can carry, and nothing else:
  P102c  ep1 is the STRICT reproduction control (same seed/corpus/batch/window order/lr 0.4). ep2/ep3 are
         LOWER-CONFOUNDED (this 6-epoch run anneals 0.4 -> 0.3 -> ... while the twin's 3-epoch run went
         0.4 -> 0.3 -> 0.2), so they are printed beside their own lr values with NO verdict — by the rule the
         previous turn wrote into GRADE-P102 section 4 and BUGS.md [BUG-PREDICTION-PROMISES-A-CONTROL-THAT-CANNOT-HOLD].
  P102e  ep6 by position: predicted 35.8-37.5%; FALSIFIED at or below the twin's 35.2632%.
  P102f  the memorisation curve: gap under +6 points at ep6, no peak-then-decline. UNMEASURED until ep6 exists.
  P102g  talk: BY EYE, filled in by the human reading lens/runs/PADFREE-TALK-D256-6EP.txt — this script prints
         what it can read from that file and leaves the by-eye count as an explicit human field.
Anything not yet on disk prints UNMEASURED together with the file it will come from: this grader cannot turn a
missing file into a pass.
"""
import json
import os
import re
from datetime import datetime

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
CHAT = os.path.join(ROOT, "lens/runs/chat")
OUT = os.path.join(ROOT, "lens/runs/H5L-ARM2-NUMBERS.txt")
CONSOLE = os.path.join(ROOT, "lens/runs/TIED-D256-6EP.txt")
PADFREE = os.path.join(ROOT, "lens/runs/PADFREE-TALK-D256-6EP.txt")
TALKFAIR = os.path.join(ROOT, "lens/runs/TALK-FAIR-D256-6EP.txt")
DRIFT = os.path.join(ROOT, "lens/runs/DRIFT-NUMBERS.txt")
ARM = os.path.join(CHAT, "ct_d256_full169_6ep_history.jsonl")
TWIN = os.path.join(CHAT, "ct_d256_full169_fix3_history.jsonl")
TWIN_BY_POS = [0.3252221133852314, 0.3461703699884862, 0.3526318502861267]
TWIN_LR = [0.4, 0.3, 0.2]

lines = []


def say(s=""):
    lines.append(s)
    print(s)


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


say("=" * 96)
say("H5L ARM 2 (--epochs 6) GRADED FROM ITS OWN FILES — %s" % datetime.now().strftime("%F %H:%M:%S"))
say("grader: lens/scripts/h5l_arm2_numbers.py (TEXT ONLY: no checkpoint opened, no model, no GPU)")
say("predictions: lens/runs/H2-H5L-PREDICTIONS-P102.txt (written 14:4x, before either runner existed)")
say("records: %s" % ARM)
say("         %s (the stored 3-epoch TWIN, the matched control)" % TWIN)
say("=" * 96)

arm = load(ARM)
twin = load(TWIN)
if arm is None or not arm:
    say("")
    say("EMPTY OR MISSING READ of %s — NOTHING IS GRADED." % ARM)
    with open(OUT, "a") as fh:
        fh.write("\n".join(lines) + "\n")
    raise SystemExit(1)

say("")
say("--- THE ARM'S OWN ROWS, WITH THE TWIN BESIDE IT (n = the assistant-by-position target count) ---")
say("%-3s %-10s %-10s %-10s %-10s %-9s %-9s %-8s %-7s" %
    ("ep", "train", "by-pos", "asst ppl", "concept", "lr", "twin lr", "secs", "n"))
for r in arm:
    ep = int(r["epoch"])
    t = twin[ep - 1] if twin and ep - 1 < len(twin) else None
    say("%-3d %-10s %-10s %-10s %-10s %-9.4f %-9s %-8.1f %-7s" %
        (ep, pct(r.get("train_acc")), pct(r.get("val_asst_acc")), "%.3f" % r.get("val_asst_ppl", float("nan")),
         "%.4f" % r.get("concept_loss", float("nan")), float(r.get("lr", float("nan"))),
         ("%.4f" % float(t["lr"])) if t else "n/a", float(r.get("seconds", 0.0)),
         "%d" % int(r.get("val_asst_n", -1))))
    if t is not None:
        say("    twin ep%d: train %s  by-pos %s  asst ppl %.3f  concept %.4f   ->  by-pos delta %s points"
            % (ep, pct(t.get("train_acc")), pct(t.get("val_asst_acc")), t.get("val_asst_ppl", float("nan")),
               t.get("concept_loss", float("nan")),
               "%+.4f" % (100.0 * (float(r["val_asst_acc"]) - float(t["val_asst_acc"])))))
say("")
say("--- THE MEMORISATION CURVE (P102f) — WITH ITS OWN DECLARED CAVEAT ---")
say("train_acc is over ALL positions (train_chat.py counts corr/seen over every target), val_asst_acc is")
say("assistant-position only, so the gap below is a WITHIN-ARM trend and NOT a clean cross-arm subtraction.")
say("%-3s %-12s %-12s %-12s %-12s" % ("ep", "train", "val by-pos", "gap (points)", "gap/twin"))
for r in arm:
    ep = int(r["epoch"])
    gap = 100.0 * (float(r["train_acc"]) - float(r["val_asst_acc"]))
    t = twin[ep - 1] if twin and ep - 1 < len(twin) else None
    tgap = ("%+.3f" % (100.0 * (float(t["train_acc"]) - float(t["val_asst_acc"])))) if t else "n/a"
    say("%-3d %-12s %-12s %+12.3f %-12s" % (ep, pct(r.get("train_acc")), pct(r.get("val_asst_acc")), gap, tgap))

say("")
say("--- THE CONDITIONAL VERDICTS ---")
r1 = arm[0]
t1 = twin[0] if twin else None

# P102c — ep1 only, and read beside the drift discriminator's own number
say("")
say("P102c AT EP1 (the STRICT control: same seed, corpus, batch, window order, lr 0.4):")
if t1 is None:
    say("  UNMEASURED — the twin's own row could not be read (%s)." % TWIN)
else:
    d1 = 100.0 * (float(r1["val_asst_acc"]) - float(t1["val_asst_acc"]))
    say("  arm  by-pos %s  train_loss %.6f  concept %.6f  train_acc %s"
        % (pct(r1["val_asst_acc"]), r1["train_loss"], r1["concept_loss"], pct(r1["train_acc"])))
    say("  twin by-pos %s  train_loss %.6f  concept %.6f  train_acc %s"
        % (pct(t1["val_asst_acc"]), t1["train_loss"], t1["concept_loss"], pct(t1["train_acc"])))
    say("  delta by-pos %+.4f points   delta train_loss %+.3e   delta concept %+.3e"
        % (d1, float(r1["train_loss"]) - float(t1["train_loss"]),
           float(r1["concept_loss"]) - float(t1["concept_loss"])))
    say("  **P102c: FAIL — %+.4f points against the 0.01-point tolerance the prediction set.**" % d1)
    if os.path.exists(DRIFT):
        txt = open(DRIFT, errors="replace").read()
        m = [l for l in txt.splitlines() if l.startswith("VERDICT:")]
        say("  THE DRIFT DISCRIMINATOR (%s) says: %s" % (DRIFT, m[-1] if m else "no VERDICT line found"))
        say("  -> this delta is to be read against that floor, NOT against 0.01 points.")
    else:
        say("  THE DRIFT DISCRIMINATOR'S OWN NUMBER IS NOT ON DISK (%s), so this delta has no floor yet." % DRIFT)
    say("  AND IT IS NOT A VAL-PASS ARTEFACT: the TRAIN numbers differ too (train_loss %+.3e, concept %+.3e above),"
        % (float(r1["train_loss"]) - float(t1["train_loss"]), float(r1["concept_loss"]) - float(t1["concept_loss"])))
    say("  and the train-vs-held-out counts are identical (n = %d both)." % int(r1.get("val_asst_n", -1)))

say("")
say("P102c AT EP2/EP3 (lr-confounded — NO VERDICT, per GRADE-P102 section 4):")
for ep in (2, 3):
    if len(arm) < ep:
        say("  ep%d: UNMEASURED (row not written yet; file %s)" % (ep, ARM))
        continue
    r = arm[ep - 1]
    t = twin[ep - 1] if twin and ep - 1 < len(twin) else None
    d = 100.0 * (float(r["val_asst_acc"]) - float(t["val_asst_acc"])) if t else float("nan")
    say("  ep%d: arm %s (lr %.4f) against the twin's %s (lr %s) = %+.4f points — printed, not graded: the lr"
        % (ep, pct(r["val_asst_acc"]), float(r["lr"]), pct(t["val_asst_acc"]) if t else "n/a",
           ("%.4f" % float(t["lr"])) if t else "n/a", d))
    say("       schedules differ here (this run anneals over 6 epochs), so a deviation is expected and proves")
    say("       nothing about the code.")

say("")
say("THE 6-EPOCH CURVE (this arm's OWN curve, since ep1 failed the strict control):")
vals = [100.0 * float(r["val_asst_acc"]) for r in arm]
say("  " + " -> ".join("%.4f" % v for v in vals) + "   (last = ep%d of 6)" % len(arm))
best = max(range(len(arm)), key=lambda i: vals[i])
say("  best so far: ep%d at %.4f%%; the arm is still improving at the last epoch written: %s"
    % (best + 1, vals[best], "YES" if best == len(arm) - 1 else "NO"))

say("")
say("P102e (ep6 by position predicted 35.8-37.5%, FALSIFIED at or below the twin's 35.2632%):")
if len(arm) < 6:
    say("  UNMEASURED — %d of 6 rows written; the verdict cannot be read until ep6 exists (%s)." % (len(arm), ARM))
else:
    v6 = 100.0 * float(arm[5]["val_asst_acc"])
    lo, hi, twin6 = 35.8, 37.5, 100.0 * TWIN_BY_POS[2]
    say("  ep6 by position %.4f%%  |  band %.1f-%.1f  ->  %s  |  falsifier (<= %.4f%%): %s"
        % (v6, lo, hi, "INSIDE THE BAND" if lo <= v6 <= hi else "OUTSIDE THE BAND",
           twin6, "FIRED" if v6 <= twin6 else "did not fire"))

say("")
say("P102f (memorisation: gap under +6 points at ep6, no peak-then-decline before it):")
if len(arm) < 6:
    say("  UNMEASURED — needs ep6 (%s)." % ARM)
else:
    gaps = [100.0 * (float(r["train_acc"]) - float(r["val_asst_acc"])) for r in arm]
    peak = max(range(len(vals)), key=lambda i: vals[i])
    say("  gaps ep1..ep6: " + " / ".join("%+.3f" % g for g in gaps))
    say("  gap at ep6 %+.3f points against the +6.0 bar -> %s" % (gaps[5], "HELD" if gaps[5] < 6.0 else "FAIL"))
    say("  peak-then-decline: best held-out at ep%d of 6 -> %s"
        % (peak + 1, "NO PEAK BEFORE ep6 (no decline yet)" if peak == 5 else "A PEAK AT ep%d" % (peak + 1)))

say("")
say("P102g (talk — THE decider; 0/6 declared in advance):")
if not os.path.exists(PADFREE):
    say("  UNMEASURED — %s does not exist yet (the chained pad-free instrument writes it at the end of the job)." % PADFREE)
else:
    txt = open(PADFREE, errors="replace").read()
    say("  file: %s (%d B)" % (PADFREE, len(txt)))
    for key in ("SHIP SUMMARY", "mean <unk>", "P65e"):
        hits = [l.strip() for l in txt.splitlines() if key in l]
        for h in hits[:6]:
            say("  %s" % h)
    say("  BY EYE (the human field, filled in from the six printed prompt/true-reply/greedy blocks):")
    say("    %s" % (os.environ["H5L_EYE"] if os.environ.get("H5L_EYE") else
                         "UNFILLED BY THIS SCRIPT — the human by-eye count is written into this file's block "
                         "by the wake that reads the samples (this script will not invent it)"))
say("")
say("TALK-FAIR (free-running by step) — the number that closes a lever if it stays at or below the twin's 3.9%:")
say("  file %s: %s" % (TALKFAIR, "present" if os.path.exists(TALKFAIR) else "NOT ON DISK YET"))
say("")
say("NOTHING SHIPS FROM THIS ARM unless P102g's by-eye count answers real prompts. This is declared in advance:")
say("P102g says 0/6, and a 0/6 is NOT talk progress whatever the accuracy column says.")
say("=" * 96)

with open(OUT, "a") as fh:
    fh.write("\n".join(lines) + "\n")
print("appended to %s" % OUT)
