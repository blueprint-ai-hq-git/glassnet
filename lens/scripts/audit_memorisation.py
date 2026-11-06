"""audit_memorisation.py — one table over every stored run: train acc vs held-out by position, per epoch.

Reads only:
  lens/runs/chat/*_history.jsonl   (the trainer's own per-epoch rows)
  lens/runs/chat/*_best.npz        (existence + mtime only, never opened)
Writes only: lens/runs/MEMORISATION-AUDIT.txt
Predictions: lens/runs/MEMORISATION-AUDIT-PREDICTIONS-P93.txt (P93a-d), written before this ran.
"""
import glob
import json
import os
import time

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
CHAT = os.path.join(PROJ, "lens", "runs", "chat")
OUT = os.path.join(PROJ, "lens", "runs", "MEMORISATION-AUDIT.txt")


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def pct(x):
    return "   n/a" if x is None else "{:6.2f}%".format(100.0 * x)


# data_dir is NOT in the history rows; the trainer records it in <tag>_log.json (train_chat.py line ~1285).
DATADIR = {}
for lp in glob.glob(os.path.join(CHAT, "*_log.json")):
    tag = os.path.basename(lp)[: -len("_log.json")]
    try:
        with open(lp) as fh:
            DATADIR[tag] = (json.load(fh).get("args") or {}).get("data_dir", "?")
    except (json.JSONDecodeError, OSError):
        DATADIR[tag] = "unreadable"


runs = []
for h in sorted(glob.glob(os.path.join(CHAT, "*_history.jsonl"))):
    tag = os.path.basename(h)[: -len("_history.jsonl")]
    rows = load(h)
    if not rows:
        continue
    # keep only rows that actually carry an epoch number and the two columns
    rows = [r for r in rows if isinstance(r.get("epoch"), int) and "val_asst_acc" in r]
    if not rows:
        continue
    runs.append((tag, h, rows))

lines = []
w = lines.append
w("=" * 118)
w("MEMORISATION AUDIT — every stored run, from its own history rows.  generated %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
w("train acc and held-out BY POSITION are the trainer's own printed columns; 'gap' = train - held-out.")
w("predictions: lens/runs/MEMORISATION-AUDIT-PREDICTIONS-P93.txt (P93a-d)")
w("=" * 118)

summary = []
no_hist = sorted(os.path.basename(p)[: -len("_best.npz")]
                 for p in glob.glob(os.path.join(CHAT, "*_best.npz"))
                 if not os.path.exists(p[: -len("_best.npz")] + "_history.jsonl"))
for tag, h, rows in runs:
    bestpath = os.path.join(CHAT, tag + "_best.npz")
    has_best = os.path.exists(bestpath)
    argmax = max(rows, key=lambda r: r["val_asst_acc"])["epoch"]
    last = rows[-1]
    w("")
    w("--- %s   (%d epoch rows on disk%s)" % (tag, len(rows), ", merged from more than one launch" if len(rows) > 1 and len({r.get("data_dir") for r in rows}) > 1 else ""))
    w("    epoch   train acc   heldout BY-POSITION   heldout legacy by-id     gap(train-heldout)   steps")
    for r in rows:
        tr = r.get("train_acc")
        gap = "   n/a" if tr is None else "{:7.2f} pt".format(100.0 * (tr - r["val_asst_acc"]))
        w("    %5d   %s        %s              %s            %s      %s"
          % (r["epoch"], pct(tr), pct(r.get("val_asst_acc")), pct(r.get("val_masked_acc")), gap,
             r.get("opt_steps", r.get("steps", "?"))))
    have = [r for r in rows if r.get("train_acc") is not None]
    tr_end = last.get("train_acc")
    note = ""
    if has_best:
        note = ("argmax epoch=%d (%.2f%%)  best.npz mtime=%s"
                % (argmax, 100.0 * max(r["val_asst_acc"] for r in rows),
                   time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(bestpath)))))
    else:
        note = "NO *_best.npz on disk for this tag"
    w("    %s" % note)
    gap_end = None if tr_end is None else tr_end - last["val_asst_acc"]
    gap2 = None
    e2 = [r for r in rows if r["epoch"] == 2]
    if e2 and e2[0].get("train_acc") is not None:
        gap2 = e2[0]["train_acc"] - e2[0]["val_asst_acc"]
    summary.append(dict(tag=tag, n=len(rows), argmax=argmax, best_acc=max(r["val_asst_acc"] for r in rows),
                        last_ep=last["epoch"], gap_end=gap_end, gap2=gap2, has_best=has_best,
                        data_dir=DATADIR.get(tag, "?"), nrows_raw=len(rows)))

w("")
w("=" * 118)
w("P93a  runs with >=3 epochs whose held-out max is at or before epoch 3:")
multi = [s for s in summary if s["n"] >= 3]
longer = [s for s in multi if s["n"] > 3]
early = [s for s in multi if s["argmax"] <= 3]
if not longer:
    w("      VACUOUS / NOT TESTABLE: every run with a history file on disk has at most 3 epochs, so 'peaks by")
    w("      epoch 3' is true of all of them by construction and says nothing. Recorded as NOT TESTABLE, not PASS.")
    w("      The only longer run in the project is ct_d256_6M_6ep (6 epochs, NO history file - its epoch lines")
    w("      are in lens/runs/TIED-FULLCORPUS.txt lines 71-81): it PEAKED AT EPOCH 5 of 6 (35.82% by its own")
    w("      legacy by-id column), then fell 0.09 pt at ep6 while train acc rose 30.03% -> 30.19% and val ppl")
    w("      rose 8,197.87 -> 10,201.15. That is the shape to expect from a longer run: late epochs stop paying.")
else:
    w("      %d of %d runs with >3 epochs peak at or before epoch 3 -> %s"
      % (len(early), len(longer), "PASS" if len(early) >= 3 else "FAIL"))
for s in multi:
    w("        %-46s argmax ep%-2d of %d    last ep%-2d   best %s" % (s["tag"], s["argmax"], s["n"], s["last_ep"], pct(s["best_acc"])))

w("")
w("P93b  argmax epoch per run (the file a best-epoch checkout must match):")
for s in summary:
    w("        %-46s argmax ep%-2d  best %s  best.npz=%s" % (s["tag"], s["argmax"], pct(s["best_acc"]), "yes" if s["has_best"] else "NO"))

w("")
w("P93c  gap at epoch 2 vs gap at the last epoch, runs with >=4 epochs:")
c_rows = [s for s in summary if s["n"] >= 4 and s["gap2"] is not None and s["gap_end"] is not None]
if not c_rows:
    w("        none on disk with >=4 epochs -> NOT TESTABLE")
else:
    ok = all(s["gap_end"] > s["gap2"] for s in c_rows)
    for s in c_rows:
        w("        %-46s ep2 %+7.2f pt -> last ep%-2d %+7.2f pt   %s"
          % (s["tag"], 100 * s["gap2"], s["last_ep"], 100 * s["gap_end"],
             "grew" if s["gap_end"] > s["gap2"] else "SHRANK"))
    w("        -> %s" % ("PASS" if ok else "FAIL"))

w("")
w("P93d  d=512 vs d=256 full-corpus final gap:")
d512 = [s for s in summary if "d512" in s["tag"] and s["gap_end"] is not None]
d256 = [s for s in summary if "d256_full" in s["tag"] and s["gap_end"] is not None and "ours" not in s["tag"]]
for s in d512 + d256:
    w("        %-46s last ep%-2d gap %+7.2f pt  (held-out %s)" % (s["tag"], s["last_ep"], 100 * s["gap_end"], pct(s["best_acc"])))
if d512 and d256:
    a = max(d512, key=lambda s: s["best_acc"])
    b = max(d256, key=lambda s: s["best_acc"])
    w("        -> %s (d512 %+0.2f vs d256 %+0.2f pt)"
      % ("PASS" if a["gap_end"] > b["gap_end"] else "FAIL", 100 * a["gap_end"], 100 * b["gap_end"]))
else:
    w("        -> NOT TESTABLE (a side is missing)")

w("")
w("NOTE ON SCOPE: a row is only comparable to another row from the same data_dir. data_dir per run is printed")
w("below; runs on the default 'data/chat/prepared' (no window cap, vocab 12,000) are NOT comparable to runs on")
w("'data/chat/prepared_mix_ctrl_honest' (vocab 12,002, val 87,035).")
for s in summary:
    w("        %-46s data_dir=%s  rows=%d" % (s["tag"], s["data_dir"], s["nrows_raw"]))
w("")
w("RUNS WITH A *_best.npz BUT NO *_history.jsonl (their per-epoch rows exist only in a text log, so any")
w("'per-epoch' claim about them must name that log):")
for t in no_hist:
    w("        %s" % t)
w("=" * 118)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
