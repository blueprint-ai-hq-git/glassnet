#!/usr/bin/env python3
"""grade_attn_full.py — grade P82a-d (and the P88a/P88b control) from the stored text of the attention arm.

READS ONLY FILES: lens/runs/ATTN-FULL.txt (the arm's log), lens/runs/chat/ct_d256_full169_attn_history.jsonl
(the arm's per-epoch records), lens/runs/chat/ct_d256_full169_fix3_history.jsonl (the COMPARATOR twin's records),
lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt (the twin's stored epoch line) and, if present,
lens/runs/ATTN-VRAM-SAMPLE.txt (peak GPU MiB sampled by the runner while the arm trained).
IT OPENS NO CHECKPOINT, LOADS NO MODEL, TOUCHES NO GPU. It writes ONLY lens/runs/GRADE-P82.txt.

Rule this file is written to obey (this project's own failure mode): every verdict prints the values it compared,
side by side, before the verdict word. Predictions being graded: lens/runs/ATTN-FULL-PREDICTIONS-P82.txt
(P82a-d, written before the arm was queued) and lens/runs/ATTN-GRADE-TALK-PREDICTIONS-P88.txt (P88a/P88b).
"""
import json
import os
import re
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"
LOG = os.path.join(P, "lens/runs/ATTN-FULL.txt")
HIST = os.path.join(P, "lens/runs/chat/ct_d256_full169_attn_history.jsonl")
TWIN_HIST = os.path.join(P, "lens/runs/chat/ct_d256_full169_fix3_history.jsonl")
TWIN_LOG = os.path.join(P, "lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt")
VRAM = os.path.join(P, "lens/runs/ATTN-VRAM-SAMPLE.txt")
OUT = os.environ.get("GRADE_OUT", os.path.join(P, "lens/runs/GRADE-P82.txt"))

RE_EP = re.compile(
    r"^\s*ep\s+(\d+)/(\d+)\s+loss\s+([-\d.eE+]+)\s+concept\s+([-\d.eE+]+)\s+acc\s+([\d.]+)%\s+"
    r"VAL\s+ppl\s+([\d.]+)\s+acc\s+([\d.]+)%\s+MASKED\(legacy by-id\)\s+ppl\s+([\d.]+)\s+acc\s+([\d.]+)%\s+"
    r"\(n=(\d+)\)\s+ASST\(by-position\)\s+ppl\s+([\d.]+)\s+acc\s+([\d.]+)%\s+\(n=(\d+)\)")
RE_EXIT = re.compile(r"^\s*exit=(-?\d+)")
RE_START = re.compile(r"^=== THE REAL ATTENTION ARM \(P82a-f\) (\d{4}-\d\d-\d\d \d\d:\d\d)")
RE_DONE = re.compile(r"^=== DONE (\d{4}-\d\d-\d\d \d\d:\d\d)")

_rows = []


def log(s=""):
    print(s, flush=True)
    _rows.append(str(s))


def read(path, what):
    if not os.path.exists(path):
        log(f"  !! MISSING {what}: {path}")
        return None
    return open(path, encoding="utf-8", errors="replace").read()


def hist_rows(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------- the arm's own log
txt = read(LOG, "the arm's log")
if txt is None:
    sys.exit(2)

epochs = []
for line in txt.splitlines():
    m = RE_EP.match(line)
    if m:
        epochs.append(dict(ep=int(m.group(1)), total=int(m.group(2)), loss=float(m.group(3)),
                           concept=float(m.group(4)), train_acc=float(m.group(5)) / 100.0,
                           val_ppl=float(m.group(6)), val_acc=float(m.group(7)) / 100.0,
                           legacy_ppl=float(m.group(8)), legacy_acc=float(m.group(9)) / 100.0,
                           legacy_n=int(m.group(10)), asst_ppl=float(m.group(11)),
                           asst_acc=float(m.group(12)) / 100.0, asst_n=int(m.group(13))))
exit_codes = [int(m.group(1)) for m in (RE_EXIT.match(ln) for ln in txt.splitlines()) if m]
start_m = RE_START.search(txt)
done_m = RE_DONE.search(txt)

hist = hist_rows(HIST)
twin_hist = hist_rows(TWIN_HIST)
twin_ep3 = twin_hist[-1] if twin_hist else None
twin_line = ""
for line in (read(TWIN_LOG, "twin log") or "").splitlines():
    if RES := re.search(r"ep 3/3 .*ASST\(by-position\).*", line):
        twin_line = RES.group(0).strip()

log("=" * 110)
log("GRADE OF P82a-d — the attention arm, read from its own stored log and records")
log("predictions being graded: lens/runs/ATTN-FULL-PREDICTIONS-P82.txt (P82a-d)")
log("instrument: lens/scripts/grade_attn_full.py (no checkpoint opened, no model, no GPU, no training)")
log("=" * 110)
log("")
log(f"arm log      : {LOG}")
log(f"arm records  : {HIST}")
log(f"twin records : {TWIN_HIST}")
log(f"twin epoch line (quoted): {twin_line or '(not found)'}")
log("")
log(f"EPOCH LINES FOUND IN THE ARM'S LOG: {len(epochs)}  (the arm runs 3)")
log("  ep   train acc   val ppl   val acc   legacy(by-id) ppl/acc        asst(BY POSITION) ppl/acc     twin asst  delta")
for e in epochs:
    tw = next((h["val_asst_acc"] for h in twin_hist if h["epoch"] == e["ep"]), None)
    if tw is None:
        twincol, deltacol = "   -    ", "    -    "
    else:
        twincol = f"{tw*100:6.2f}%"
        deltacol = f"{(e['asst_acc']-tw)*100:+6.2f} pt"
    log(f"  {e['ep']}/{e['total']}  {e['train_acc']*100:8.2f}%  {e['val_ppl']:8.2f}  {e['val_acc']*100:7.2f}%"
        f"   {e['legacy_ppl']:8.2f} / {e['legacy_acc']*100:6.2f}% (n={e['legacy_n']})"
        f"   {e['asst_ppl']:8.2f} / {e['asst_acc']*100:6.2f}% (n={e['asst_n']})   {twincol}  {deltacol}")
log("")

# ---------------------------------------------------------------- P82a / P88a control
log("-" * 110)
log("P88a / P82a CONTROL — the grader against the arm's own per-epoch records, and the run's shape")
log("-" * 110)
ok = True
if len(hist) != len(epochs):
    ok = False
    log(f"  COUNT MISMATCH: {len(epochs)} epoch lines in the log against {len(hist)} records in the jsonl")
for e, h in zip(epochs, hist):
    pairs = [("train_acc", e["train_acc"], h["train_acc"]), ("asst_acc", e["asst_acc"], h["val_asst_acc"]),
             ("legacy_acc", e["legacy_acc"], h["val_masked_acc"])]
    for name, a, b in pairs:
        same = abs(a - b) < 5e-5
        if not same:
            ok = False
        log(f"  ep{e['ep']} {name:11s} log {a*100:7.2f}%   jsonl {b*100:7.2f}%   -> {'MATCH' if same else 'DIFFER'}")
log(f"  -> P88a {'PASS' if ok else 'FAIL'} (all {3*len(epochs)} printed numbers reproduced from the records)")
log("")
three = len(epochs) == 3
finite = all(abs(e["loss"]) < 1e6 and abs(e["train_acc"]) < 1e6 for e in epochs)
log(f"  epoch lines: {len(epochs)} of 3   losses finite: {finite}   runner exit= lines: {exit_codes}")
if exit_codes:
    log(f"  exit code(s): {exit_codes}   (0 = clean)")
for e in epochs:
    log(f"  ep{e['ep']} loss {e['loss']:.4f} concept {e['concept']:.4f} asst {e['asst_acc']*100:.2f}%")
p82a = three and finite and (not exit_codes or exit_codes[-1] == 0)
log(f"  -> P82a {'PASS' if p82a else 'FAIL'}  (three epoch lines, all finite, clean exit)")
log("")

# ---------------------------------------------------------------- P82b cost
log("-" * 110)
log("P82b — COST: wall between 1.0 h and 2.5 h for 3 epochs, peak GPU memory below 2.5 GB")
log("-" * 110)
secs = [h.get("seconds") for h in hist if h.get("seconds")]
if secs:
    log(f"  per-epoch seconds from the records: {[round(s) for s in secs]}  (sum {sum(secs)/3600:.2f} h)")
if start_m and done_m:
    from datetime import datetime
    t0 = datetime.strptime(start_m.group(1), "%Y-%m-%d %H:%M")
    t1 = datetime.strptime(done_m.group(1), "%Y-%m-%d %H:%M")
    wall = (t1 - t0).total_seconds() / 3600.0
    log(f"  runner wall from its own timestamps: {start_m.group(1)} -> {done_m.group(1)} = {wall:.2f} h")
elif secs:
    # RACE, seen on the first pass (2026-10-23 21:0x): the waiter stops as soon as the PYTHON process is gone,
    # and the bash wrapper writes its '=== DONE' line a moment later, so the grader can run before it exists.
    # Fallback, clearly labelled: the sum of the per-epoch seconds, which is the training time without the
    # data build. Both numbers are printed if both are available.
    wall = sum(secs) / 3600.0
    log(f"  runner start/done lines: start={bool(start_m)} done={bool(done_m)} -- DONE not yet written;"
        f" wall taken from the SUM OF PER-EPOCH SECONDS = {wall:.2f} h (training only, no data build)")
else:
    wall = None
    log(f"  runner start/done lines: start={bool(start_m)} done={bool(done_m)} (the arm may still be running)")
peak = None
if os.path.exists(VRAM):
    try:
        peak = int(open(VRAM).read().strip().split()[0])
        log(f"  peak GPU MiB sampled by the runner while the arm trained: {peak} MiB "
            f"(file {os.path.basename(VRAM)}; nvidia-smi sampled every 15 s)")
    except Exception as ex:                                     # noqa: BLE001
        log(f"  VRAM sample file unreadable ({ex}); no peak reported")
else:
    log("  no VRAM sample file yet; peak not measured (NOT claimed as under the bar)")
w_ok = (wall is not None and 1.0 <= wall <= 2.5)
v_ok = (peak is not None and peak < 2500)
if wall is None or peak is None:
    verdict = "INCONCLUSIVE (one side unmeasured)"
else:
    verdict = "PASS" if (w_ok and v_ok) else "FAIL"
log(f"  -> P82b {verdict}   (wall bar 1.0-2.5 h {'' if wall is None else f'[{wall:.2f} h]'};"
    f" VRAM bar < 2500 MiB {'' if peak is None else f'[{peak} MiB]'})")
log("")

# ---------------------------------------------------------------- P82c / P88b held-out
log("-" * 110)
log("P82c / P88b — HELD-OUT DOES NOT COLLAPSE: ep3 by-position within 3.0 points of the twin's 35.26%")
log("-" * 110)
tw3 = twin_ep3["val_asst_acc"] if twin_ep3 else 0.3526
lo, hi = tw3 - 0.03, tw3 + 0.03
log(f"  twin ep3 by-position (from ct_d256_full169_fix3_history.jsonl): {tw3*100:.2f}%")
if twin_ep3:
    log(f"    the same record's other columns: legacy by-id {twin_ep3['val_masked_acc']*100:.2f}%,"
        f" train {twin_ep3['train_acc']*100:.2f}%")
    log(f"    quoted epoch line: {twin_line}")
log(f"  P82c band: {lo*100:.2f}% .. {hi*100:.2f}%  (= {tw3*100:.2f} +- 3.00 points)")
if len(epochs) >= 3:
    e3 = epochs[2]
    inside = lo <= e3["asst_acc"] <= hi
    log(f"  arm ep3 by-position {e3['asst_acc']*100:.2f}%  against the band -> {'INSIDE' if inside else 'OUTSIDE'}"
        f"  ({'PASS' if inside else 'FALSIFIED, BELOW' if e3['asst_acc'] < lo else 'FALSIFIED, ABOVE'})"
        f"   delta against the twin: {(e3['asst_acc']-tw3)*100:+.2f} points")
    log(f"  -> P82c {'PASS' if inside else 'FALSIFIED'}")
    log(f"  P88b predicted 31.9% +- 1.0 -> measured {e3['asst_acc']*100:.2f}% : "
        f"{'HELD' if abs(e3['asst_acc']*100 - 31.9) <= 1.0 else 'FAILED'}"
        f"  (P88b said 'if ep3 >= 32.3 I was wrong about the direction')")
else:
    log(f"  only {len(epochs)} epoch(s) in the log: P82c NOT YET GRADABLE")
log("")

# ---------------------------------------------------------------- P82d memorisation gap
log("-" * 110)
log("P82d — THE MEMORISATION GAP, COMPARED (train minus held-out by position), not celebrated")
log("-" * 110)
log(f"  {'ep':>3}  {'arm train':>10}  {'arm by-position':>16}  {'arm gap':>9}   {'twin train':>10}  "
    f"{'twin by-pos':>12}  {'twin gap':>9}")
for e in epochs:
    tw = next((h for h in twin_hist if h["epoch"] == e["ep"]), None)
    if tw:
        log(f"  {e['ep']:>3}  {e['train_acc']*100:9.2f}%  {e['asst_acc']*100:15.2f}%  "
            f"{(e['train_acc']-e['asst_acc'])*100:+8.2f}   {tw['train_acc']*100:9.2f}%  "
            f"{tw['val_asst_acc']*100:11.2f}%  {(tw['train_acc']-tw['val_asst_acc'])*100:+8.2f}")
    else:
        log(f"  {e['ep']:>3}  {e['train_acc']*100:9.2f}%  {e['asst_acc']*100:15.2f}%  "
            f"{(e['train_acc']-e['asst_acc'])*100:+8.2f}   (no twin record)")
log("  reading rule this project uses: a gap that GROWS across epochs is memorisation; this arm's twin was"
    " +3.31 points at ep3 (its best epoch) and the 12-epoch 1.5M-window tied run was +16.8.")
log("")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(_rows) + "\n")
print(f"\nwritten to {OUT}")
