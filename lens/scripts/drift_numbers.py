#!/usr/bin/env python3
"""DRIFT DISCRIMINATOR grader — TEXT ONLY. No checkpoint is opened, no model is built, no GPU is touched.

Reads the two rows the drift runs wrote (RUN A / RUN B, identical argv except --out) plus the stored reference
rows on this machine, prints every number BESIDE the file it came from, and applies the verdict rule that was
written BEFORE the runs existed (P107b in lens/runs/DRIFT-DISCRIMINATOR-PREDICTIONS-P107.txt).

    A-vs-B differ at all           -> cause (ii): the machine is not run-to-run reproducible; |d| A-vs-B is the
                                      floor under every small delta in this project.
    A == B, twin not reproduced    -> cause (i): a code-path / RNG-consumption change since the twin ran.
    A == B == twin                 -> the twin gap belongs to the --epochs 6 run specifically.
    A.n != 58,191 or a missing row -> WIRING-ONLY (a tiny-corpus pre-flight): NO VERDICT, by construction.

Env: DRIFT_TAG_A, DRIFT_TAG_B (defaults drift_rerun_a / drift_rerun_b), DRIFT_REF (label=path;label=path),
     DRIFT_LOG (default lens/runs/DRIFT-DISCRIMINATOR.txt) for the parameters line.
Writes lens/runs/DRIFT-NUMBERS.txt (its own block; the runner's log gets the same text on stdout).
"""
import json
import os
import re
import sys
from datetime import datetime

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
CHAT = os.path.join(ROOT, "lens/runs/chat")
OUT = os.path.join(ROOT, "lens/runs/DRIFT-NUMBERS.txt")
LOGP = os.environ.get("DRIFT_LOG", os.path.join(ROOT, "lens/runs/DRIFT-DISCRIMINATOR.txt"))

TAG_A = os.environ.get("DRIFT_TAG_A", "drift_rerun_a")
TAG_B = os.environ.get("DRIFT_TAG_B", "drift_rerun_b")
REFS = os.environ.get(
    "DRIFT_REF",
    "TWIN ct_d256_full169_fix3 (2026-10-22 18:27, 3 ep, --epochs 3)="
    + os.path.join(CHAT, "ct_d256_full169_fix3_history.jsonl")
    + ";6EP arm ct_d256_full169_6ep (2026-10-25 15:42, --epochs 6)="
    + os.path.join(CHAT, "ct_d256_full169_6ep_history.jsonl"),
)

FIELDS = ["train_loss", "concept_loss", "train_acc", "val_ppl", "val_acc",
          "val_masked_acc", "val_masked_ppl", "val_asst_acc", "val_asst_ppl", "val_asst_n"]
ACC_FIELDS = {"train_acc", "val_acc", "val_masked_acc", "val_asst_acc"}

lines = []


def say(s=""):
    lines.append(s)
    print(s)


def row_of(path, ep=1):
    if not os.path.exists(path):
        return None, "MISSING FILE"
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    if not rows:
        return None, "EMPTY READ"
    for r in rows:
        if int(r.get("epoch", -1)) == ep:
            return r, None
    return None, "NO EPOCH %d ROW" % ep


def fmt(field, v):
    if v is None:
        return "n/a"
    if field in ACC_FIELDS:
        return "%.4f%%" % (100.0 * float(v))
    if field == "val_asst_n":
        return "%d" % int(v)
    return "%.6g" % float(v)


def delta(field, va, vb):
    if va is None or vb is None:
        return "n/a"
    d = float(va) - float(vb)
    if field in ACC_FIELDS:
        return "%+.4f points" % (100.0 * d)
    if field == "val_asst_n":
        return "%+d" % (int(va) - int(vb))
    return "%+.3e" % d


def params_line(path):
    """the trainer's own header line, so the parameter count is read from a file rather than assumed"""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, errors="replace") as fh:
        for ln in fh:
            m = re.search(r"model:\s*([\d,]+)\s*params\s*\(([^)]*)\)", ln)
            if m:
                out.append(m.group(0).strip())
    return out


say("=" * 96)
say("DRIFT NUMBERS — %s" % datetime.now().strftime("%F %H:%M:%S"))
say("grader: lens/scripts/drift_numbers.py (TEXT ONLY — no checkpoint opened, no model, no GPU)")
say("predictions: lens/runs/DRIFT-DISCRIMINATOR-PREDICTIONS-P107.txt (written before the runner existed)")
say("=" * 96)

pa = os.path.join(CHAT, "%s_history.jsonl" % TAG_A)
pb = os.path.join(CHAT, "%s_history.jsonl" % TAG_B)
say("")
say("RUN A file: %s" % pa)
say("RUN B file: %s" % pb)
say("  (the two runs' argv is identical EXCEPT --out; see the configuration lines above in the runner's log)")

A, ea = row_of(pa)
B, eb = row_of(pb)
if ea or eb:
    say("")
    say("READ FAILURE — NOTHING IS GRADED, NO DELTA IS PRINTED:")
    say("  RUN A: %s" % (ea or "ok"))
    say("  RUN B: %s" % (eb or "ok"))
    say("An empty or missing read is exit 1 here BY DESIGN: a clean-looking verdict over nothing is how this")
    say("project once printed a PASS over 0 rows.")
    payload = "\n".join(lines) + "\n"
    with open(OUT, "a") as fh:
        fh.write(payload)
    sys.exit(1)

say("")
say("--- RUN A vs RUN B, FIELD BY FIELD (epoch 1) ---")
say("%-16s %-22s %-22s %s" % ("field", "RUN A", "RUN B", "A - B"))
for f in FIELDS:
    say("%-16s %-22s %-22s %s" % (f, fmt(f, A.get(f)), fmt(f, B.get(f)), delta(f, A.get(f), B.get(f))))
say("%-16s %-22s %-22s %s" % ("seconds", fmt("seconds", A.get("seconds")),
                              fmt("seconds", B.get("seconds")),
                              delta("seconds", A.get("seconds"), B.get("seconds"))))

identical = all(A.get(f) == B.get(f) for f in FIELDS)
diffs = [(f, A.get(f), B.get(f)) for f in FIELDS if A.get(f) != B.get(f)]
say("")
say("A == B in every printed field: %s   (%d field(s) differ)" % (identical, len(diffs)))

say("")
say("--- THE SAME FIELDS BESIDE THE STORED ROWS ON THIS MACHINE (each with the file it came from) ---")
refrows = []
for item in [x for x in REFS.split(";") if x.strip()]:
    label, path = item.rsplit("=", 1)
    r, err = row_of(path, ep=1)
    refrows.append((label, path, r))
    say("")
    say("REFERENCE %s" % label)
    say("  file: %s" % path)
    if err:
        say("  READ: %s — this reference is NOT used" % err)
        continue
    for f in ["train_loss", "concept_loss", "train_acc", "val_asst_acc", "val_asst_ppl", "val_asst_n"]:
        say("  %-14s %-22s | A - ref %-16s | B - ref %s"
            % (f, fmt(f, r.get(f)), delta(f, A.get(f), r.get(f)), delta(f, B.get(f), r.get(f))))

plines = params_line(LOGP)
say("")
say("--- THE RUNNER'S OWN CONFIGURATION LINES (%s) ---" % LOGP)
if plines:
    for i, ln in enumerate(plines[-4:], 1):
        say("  %s" % ln)
    say("  (%d such line(s); the two runs of this file must print the same ones)" % len(plines))
else:
    say("  none found — the runner's log is missing or carries no model line; the parameter count is then")
    say("  UNMEASURED and no P107a claim is made from it.")

say("")
say("=" * 96)
say("VERDICT, BY THE RULE WRITTEN BEFORE THE RUNS (P107b)")
say("=" * 96)
nA = int(A.get("val_asst_n", -1))
nB = int(B.get("val_asst_n", -1))
twin = None
for label, path, r in refrows:
    if label.startswith("TWIN") and r is not None:
        twin = (label, path, r)

if nA != 58191 or nB != 58191:
    say("WIRING-ONLY PRE-FLIGHT: n = %d (A) and %d (B) against the full-corpus 58,191." % (nA, nB))
    say("NO P107 VERDICT IS DRAWN FROM A TINY-CORPUS RUN — by construction, and this line is why the")
    say("pre-flight's own log prints the overridden configuration in its header.")
    verdict = "WIRING-ONLY"
elif not identical:
    worst = max(abs(float(va) - float(vb)) for f, va, vb in diffs if f in ACC_FIELDS) if any(
        f in ACC_FIELDS for f, _, _ in diffs) else None
    say("A AND B DIFFER — CAUSE (ii) IS ESTABLISHED AS THE MECHANISM:")
    say("  the SAME command, run twice on this machine, does not return the same numbers. The twin gap at")
    say("  GRADE-P102 section 6 (0.0687 points of held-out accuracy, 8.68e-05 of train loss) is therefore")
    say("  machine noise of a size this run has now MEASURED on its own, and no code change needs to be")
    say("  hunted for it. Fields that differ, with their values:")
    for f, va, vb in diffs:
        say("    %-14s A %-22s B %-22s %s" % (f, fmt(f, va), fmt(f, vb), delta(f, va, vb)))
    say("  THE FLOOR: A-vs-B |d| = %s in held-out by-position accuracy, %s in train loss."
        % (("n/a" if worst is None else "%+.4f points" % (100.0 * worst)),
           ("n/a" if A.get("train_loss") is None or B.get("train_loss") is None
            else "%+.3e" % (float(A["train_loss"]) - float(B["train_loss"])))))
    say("  READ EVERY SMALL DELTA IN THIS PROJECT AGAINST THAT FLOOR: H2's +0.46 points (GRADE-P102")
    say("  section 1) and the d=512 six-epoch extension's +0.19 (GRADE-P94) are only meaningful as")
    say("  multiples of it, and a sub-floor difference is not a result.")
    verdict = "CAUSE (ii) — NOT RUN-TO-RUN REPRODUCIBLE"
elif twin is not None and all(A.get(f) == twin[2].get(f) for f in FIELDS):
    say("A == B AND BOTH REPRODUCE THE STORED TWIN'S EPOCH 1 ROW EXACTLY.")
    say("  So the command reproduces itself AND the twin: the gap reported in GRADE-P102 section 6 belongs")
    say("  to the --epochs 6 run specifically, and the next question is inside the trainer's --epochs")
    say("  handling, NOT the GPU. No further conclusion is drawn here.")
    verdict = "THIRD BRANCH — the twin gap belongs to the --epochs 6 run"
elif twin is not None:
    say("A == B BIT-IDENTICAL, BUT NEITHER REPRODUCES THE STORED TWIN'S EPOCH 1 ROW:")
    for f in ["train_loss", "concept_loss", "train_acc", "val_asst_acc", "val_asst_ppl"]:
        say("    %-14s A %-22s TWIN %-22s %s"
            % (f, fmt(f, A.get(f)), fmt(f, twin[2].get(f)), delta(f, A.get(f), twin[2].get(f))))
    say("  CAUSE (i) IS ON THE TABLE: the command IS reproducible on this machine, and the twin's row is not")
    say("  what it produces. That is a code-path / RNG-consumption change between 2026-10-22 18:27 and now,")
    say("  and it is a FINDING: chase it with the code diff against the nearest pre-twin-era copies")
    say("  (lens/runs/_tmp/train_chat_prefix_maskfix.py, _tmp/train_chat_prefix_unkdrop_20261024-075441.py,")
    say("  _tmp/_new_train_chat_check.py — noting train_chat.py's own mtime is 2026-10-25 12:54).")
    verdict = "CAUSE (i) — CODE / RNG PATH CHANGE SINCE 2026-10-22"
else:
    say("A == B, and NO TWIN REFERENCE ROW COULD BE READ, so the rule's second and third branches cannot be")
    say("told apart. The numbers are printed above; NO VERDICT is drawn. This is a read failure of the")
    say("control, not a result.")
    verdict = "NO VERDICT — twin reference unreadable"

say("VERDICT: %s" % verdict)
say("sources: A=%s | B=%s" % (pa, pb))
say("predictions were written before the runs: lens/runs/DRIFT-DISCRIMINATOR-PREDICTIONS-P107.txt")
say("NOTHING SHIPS FROM THIS RUN: it is a measurement of the instrument, not a model result.")
say("=" * 96)

with open(OUT, "a") as fh:
    fh.write("\n".join(lines) + "\n")
print("")
print("appended to %s" % OUT)
sys.exit(0)
