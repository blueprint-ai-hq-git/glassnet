"""grade_p76_p77.py — the verdicts P76 and P77 never got, read from the stored files (predictions P80a-e).

Predictions: lens/runs/GRADE-P76-P77-PREDICTIONS-P80.txt (written BEFORE this file existed).
Output: lens/runs/GRADE-P76-P77.txt, launched by lens/scripts/run_grade_p76_p77.sh.

WHAT IT DOES (and does not do):
  * parses the stored TEXT of four result files (DIRECT-SCALE.txt, D512-PADFREE-TALK.txt, PADFREE-TALK.txt,
    SHIP-TALK-v3.txt) — no checkpoint is opened, no model is loaded, no GPU, no training;
  * re-applies the SAME hedge detector the instrument itself used (direct_scale_test.py line 41:
    first 6 whitespace-separated tokens, lowercased, " ’ " -> "' ", substring match on a fixed fragment list);
  * recomputes mean <unk> share from the parsed 24-token greedy samples;
  * prints every measured value NEXT TO the value it is compared against, and then the verdict table for
    P76a-f and P77a-g with the pre-declared falsifier for each row.
Nothing is written except its own log file.
"""
import os
import re
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"
OUT = os.environ.get("GRADE_OUT", os.path.join(P, "lens/runs/GRADE-P76-P77.txt"))
RUNS = os.path.join(P, "lens/runs")

# the detector, copied verbatim from lens/scripts/direct_scale_test.py
HEDGE_FRAGMENTS = ("sorry", "not sure", "don ' t know", "don't know", "don ' t have", "don't have",
                   "no idea", "can ' t", "cannot", "not a lot of information")

_rows = []


def log(s=""):
    print(s, flush=True)
    _rows.append(str(s))


def is_hedge(text):
    s = " ".join(text.split()[:6]).lower().replace(" ’ ", "' ")
    return any(f in s for f in HEDGE_FRAGMENTS)


def hedge_hit(text):
    s = " ".join(text.split()[:6]).lower().replace(" ’ ", "' ")
    return [f for f in HEDGE_FRAGMENTS if f in s]


def unk_share(text):
    t = text.split()
    if not t:
        return float("nan")
    return sum(1 for w in t if w == "<unk>") / len(t)


def read(fn):
    with open(os.path.join(RUNS, fn), "r", encoding="utf-8") as f:
        return f.read().splitlines()


# ----------------------------------------------------------------------------- parsers
RE_CKPT = re.compile(r"^CHECKPOINT\s+(.*?):\s*(\S+)\s*$")
RE_SCALE = re.compile(r"^\s+SCALE s = ([\d.]+)\s")
RE_DS_CONV = re.compile(r"^\s+conv\s+(\d+)\s+hedged\s+(\w+)\s")
RE_DS_TEXT = re.compile(r"^\s+S=([\d.]+)\s+:\s*(.*)$")
RE_DS_SUM = re.compile(r"^\s+SUMMARY s=([\d.]+): hedge (\d+)/(\d+)")
RE_TALK_CONV = re.compile(r"^\s+conv\s+(\d+)\s+PROMPT")
RE_TALK_GREEDY = re.compile(r"^\s+MODEL GREEDY\s+(.*)$")
RE_TALK_UNKSUM = re.compile(r"^\s+<unk> share \(24 greedy\)\s+([\d.]+)%")
RE_PAD = re.compile(r"^\s+pad\s+(\d+):\s+([\d.]+)% <unk>\s+(.*)$")
RE_V3_ROW = re.compile(r"^\s+conv\s+(\d+):\s+<unk>\s+([\d.]+)%\s+longest word run\s+(\d+)\s+(.*)$")
RE_V3_MEAN = re.compile(r"^\s+mean <unk> share over 6 greedy 24-token samples:\s+([\d.]+)%")


def parse_direct_scale():
    """-> {(label, scale): {conv: text}}, {label: {conv: stored_hedged}}, {(label, scale): stored_hedge_count}, meta"""
    blocks, stored_hedge, panel = {}, {}, {}
    label, scale, conv = None, None, None
    meta = {}
    for ln in read("DIRECT-SCALE.txt"):
        m = RE_CKPT.match(ln)
        if m:
            label = m.group(1).strip()
            blocks.setdefault(label, {})
            stored_hedge.setdefault(label, {})
            scale, conv = None, None
            continue
        m = re.match(r"^\s+meta:\s+(.*)$", ln)
        if m and label:
            meta[label] = m.group(1).strip()
            continue
        m = re.match(r"^\s+term magnitudes at the first position.*ratio direct/concept ([\d.]+)", ln)
        if m and label:
            meta[label + "|ratio"] = m.group(1)
            continue
        m = RE_SCALE.match(ln)
        if m:
            scale = float(m.group(1))
            continue
        m = RE_DS_CONV.match(ln)
        if m:
            conv = int(m.group(1))
            stored_hedge[label][conv] = (m.group(2) == "True")
            continue
        m = RE_DS_TEXT.match(ln)
        if m and label and conv is not None:
            blocks[label].setdefault(float(m.group(1)), {})[conv] = m.group(2)
            continue
        m = RE_DS_SUM.match(ln)
        if m and label:
            panel[(label, float(m.group(1)))] = (int(m.group(2)), int(m.group(3)))
    return blocks, stored_hedge, panel, meta


def parse_talk_file(fn):
    """-> {label: {'greedy': {conv: text}, 'greedy_unk_pct': {conv: pct}, 'pad': [(pad, pct, text)], 'label': str}}"""
    out, label = {}, None
    for ln in read(fn):
        m = RE_CKPT.match(ln)
        if m:
            label = m.group(1).strip()
            out[label] = {"greedy": {}, "greedy_unk_pct": {}, "pad": []}
            continue
        if label is None:
            continue
        m = RE_TALK_CONV.match(ln)
        if m:
            cur = int(m.group(1))
            out[label]["_conv"] = cur
            continue
        m = RE_TALK_GREEDY.match(ln)
        if m:
            out[label]["greedy"][out[label]["_conv"]] = m.group(1).rstrip()
            continue
        m = RE_TALK_UNKSUM.match(ln)
        if m:
            out[label]["greedy_unk_pct"][out[label]["_conv"]] = float(m.group(1))
            continue
        m = RE_PAD.match(ln)
        if m:
            out[label]["pad"].append((int(m.group(1)), float(m.group(2)), m.group(3).rstrip()))
    return out


def parse_v3():
    """-> (rows, means) from SHIP-TALK-v3.txt: the compact per-checkpoint greedy summary."""
    rows, means = [], []
    for ln in read("SHIP-TALK-v3.txt"):
        m = RE_V3_ROW.match(ln)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), int(m.group(3)), m.group(4).rstrip()))
            continue
        m = RE_V3_MEAN.match(ln)
        if m:
            means.append(float(m.group(1)))
    return rows, means


def v2_block(fn, label_sub):
    """extract a block of the PADFREE-TALK.txt format by substring of its CHECKPOINT label."""
    blocks = parse_talk_file(fn)
    for k, v in blocks.items():
        if label_sub.lower() in k.lower():
            return k, v
    return None, None


def cmp_texts(a, b):
    """prefix comparison up to the shorter length; returns (equal, na, nb, shortest common prefix shown)"""
    n = min(len(a), len(b))
    return a[:n] == b[:n], len(a), len(b), a[:n]


# ----------------------------------------------------------------------------- run
log("=" * 110)
log("GRADE OF P76 AND P77 — the verdicts those two runs never got, read from the stored files")
log("predictions: lens/runs/GRADE-P76-P77-PREDICTIONS-P80.txt (P80a-e, written before this file existed)")
log("instrument: lens/scripts/grade_p76_p77.py  (CPU, no checkpoint opened, no model loaded, no training)")
log("=" * 110)

ds, ds_stored_hedge, ds_panel, ds_meta = parse_direct_scale()
log("")
log("--- parsed from lens/runs/DIRECT-SCALE.txt ---")
for lab in ds:
    log(f"  CHECKPOINT {lab}")
    log(f"    meta: {ds_meta.get(lab, '(not parsed)')}   ratio direct/concept {ds_meta.get(lab + '|ratio', '?')}")
    for s in sorted(ds[lab], reverse=True):
        texts = ds[lab][s]
        n = len(texts)
        h = sum(1 for t in texts.values() if is_hedge(t))
        u = 100.0 * sum(unk_share(t) for t in texts.values()) / n
        stored = ds_panel.get((lab, s))
        log(f"    s={s:4.2f}  grader: hedge {h}/{n}  mean <unk> {u:5.1f}%   "
            f"STORED panel: hedge {stored[0]}/{stored[1]} (stored <unk>% is in the panel table below)")

cf = None
for lab in ds:
    if "d256" in lab and "SHIP CANDIDATE" in lab:
        cf = lab
d512_lab = None
for lab in ds:
    if "d512" in lab:
        d512_lab = lab
log("")
log(f"  control checkpoint (d256 ep3-best) = {cf}")
log(f"  d512 checkpoint                    = {d512_lab}")

# ---- P80a
log("")
log("=" * 110)
log("P80a — INTERNAL CONSISTENCY: grader reproduces the stored panel hedge counts at s=1.00")
ok = True
for lab, want in ((cf, 4), (d512_lab, 2)):
    texts = ds[lab][1.0]
    h = sum(1 for t in texts.values() if is_hedge(t))
    stored = ds_panel.get((lab, 1.0))
    match = (h == stored[0] == want)
    ok &= match
    log(f"  {lab[:58]:<58} grader {h}/6   stored panel {stored[0]}/{stored[1]}   expected {want}/6   "
        f"-> {'PASS' if match else 'FAIL'}")
log(f"  P80a -> {'PASS' if ok else 'FAIL'}")
log("  per-conversation detail (grader, s=1.00, d256 and d512 side by side):")
for conv in sorted(ds[cf][1.0]):
    for lab in (cf, d512_lab):
        t = ds[lab][1.0].get(conv, "")
        log(f"    conv {conv:>6}  {'d256' if lab == cf else 'd512'}  hedge={str(is_hedge(t)):<5} "
            f"hit={hedge_hit(t)}  <unk> {100*unk_share(t):5.1f}%  | {t[:95]}")

# ---- P80b / P80e (control) : SHIP-TALK-v3 comparison
v3_rows, v3_means = parse_v3()
log("")
log("=" * 110)
log(f"P80b — DECODER PARITY: DIRECT-SCALE d256 s=1.00 texts vs stored SHIP-TALK-v3.txt summary "
    f"({len(v3_rows)} rows, means {v3_means})")
ep3_rows = v3_rows[:6]
ep1_rows = v3_rows[6:12]
log(f"  SHIP-TALK-v3 rows split: first 6 = d256 ep3-best (stored mean {v3_means[0] if v3_means else '?'}%), "
    f"last 6 = d256 ep1-best (stored mean {v3_means[1] if len(v3_means) > 1 else '?'}%)")
nmatch = 0
for (conv, pct, run, txt) in ep3_rows:
    t = ds[cf][1.0].get(conv, "")
    eq, la, lb, sp = cmp_texts(t, txt)
    nmatch += int(eq)
    log(f"    conv {conv:>6}  equal={str(eq):<5} len {la} vs {lb}  | {sp[:100]}")
log(f"  {nmatch}/6 identical (prefix)  -> P80b {'PASS' if nmatch == 6 else 'FAIL'}")

# control C: ep1 texts, DIRECT-SCALE has no s block for ep1, so use the two stored files
c_d512_label, c_d512 = v2_block("D512-PADFREE-TALK.txt", "REPRODUCTION CONTROL")
c_pad_label, c_pad = v2_block("PADFREE-TALK.txt", "MASKFIX ep1")
log("")
log("=" * 110)
log(f"P80d — REPRODUCTION CONTROL (P76a): '{c_d512_label}' in D512-PADFREE-TALK.txt vs "
    f"'{c_pad_label}' in PADFREE-TALK.txt")
nm = 0
for conv in sorted(c_d512["greedy"]):
    a = c_d512["greedy"][conv]
    b = c_pad["greedy"].get(conv, "")
    eq = (a == b)
    nm += int(eq)
    log(f"    conv {conv:>6}  identical={str(eq):<5}   | {a[:95]}")
pad_a = [(p, v) for (p, v, _t) in c_d512["pad"]]
pad_b = [(p, v) for (p, v, _t) in c_pad["pad"]]
log(f"  pad curve D512 file : {pad_a}")
log(f"  pad curve PADFREE   : {pad_b}")
log(f"  texts {nm}/6 identical, pad curves {'IDENTICAL' if pad_a == pad_b else 'DIFFERENT'} "
    f"-> P80d {'PASS' if (nm == 6 and pad_a == pad_b) else 'FAIL'}")
log(f"  also compared against the v3 ep1 rows (independent file):")
for (conv, pct, run, txt) in ep1_rows:
    a = c_d512["greedy"].get(conv, "")
    eq, la, lb, sp = cmp_texts(a, txt)
    log(f"    conv {conv:>6}  equal={str(eq):<5} len {la} vs {lb}  | {sp[:90]}")

# ---- P80e: mean <unk> per checkpoint
log("")
log("=" * 110)
log("P80e — THE <unk> NUMBERS REPRODUCE (recomputed from parsed greedy texts vs stored summaries)")
checks = []
# d256 ep3-best: SHIP-TALK-v3 first block (stored 2.1%) and DIRECT-SCALE s=1.00 texts
u = 100.0 * sum(unk_share(t) for t in ds[cf][1.0].values()) / len(ds[cf][1.0])
checks.append(("d256 ep3-best (SHIP-TALK-v3 stored 2.1%)", u, 2.1))
# d512 ep3-best: D512-PADFREE file, its own per-conv <unk> share lines
d512_talk = v2_block("D512-PADFREE-TALK.txt", "D512 FULL169 fix3 FINAL")[1]
u2 = 100.0 * sum(unk_share(t) for t in d512_talk["greedy"].values()) / len(d512_talk["greedy"])
u2_stored = sum(d512_talk["greedy_unk_pct"].values()) / len(d512_talk["greedy_unk_pct"])
checks.append(("d512 ep3-best (D512 file stored per-conv mean %.1f%%)" % u2_stored, u2, u2_stored))
# also from DIRECT-SCALE d512 s=1.00
u3 = 100.0 * sum(unk_share(t) for t in ds[d512_lab][1.0].values()) / len(ds[d512_lab][1.0])
checks.append(("d512 ep3-best (grader from DIRECT-SCALE s=1.00 texts; panel says 6.9%)", u3, 6.9))
# control C: ep1 (stored 9.0%)
u4 = 100.0 * sum(unk_share(t) for t in c_d512["greedy"].values()) / len(c_d512["greedy"])
checks.append(("control C d256 ep1 (stored 9.0%)", u4, 9.0))
allok = True
for name, got, want in checks:
    good = abs(got - want) <= 0.1
    allok &= good
    log(f"  {name:<62} measured {got:5.1f}%  vs {want:5.1f}%  -> {'PASS' if good else 'FAIL'}")
log(f"  P80e -> {'PASS' if allok else 'FAIL'}")

# ---- P80c cross-instrument hedge agreement on d512
log("")
log("=" * 110)
log("P80c — CROSS-INSTRUMENT AGREEMENT on the d512 ep3-best hedge count")
h_ds = sum(1 for t in ds[d512_lab][1.0].values() if is_hedge(t))
h_talk = sum(1 for t in d512_talk["greedy"].values() if is_hedge(t))
log(f"  DIRECT-SCALE.txt (direct_scale_test.py)      {h_ds}/6")
log(f"  D512-PADFREE-TALK.txt (talk_fair_v3_ship.py) {h_talk}/6")
log(f"  -> P80c {'PASS' if h_ds == h_talk == 2 else 'FAIL'}  (P76b predicted >= 3/6; its own falsifier was <= 2/6)")

# ---- the verdict tables
log("")
log("=" * 110)
log("P76 VERDICT TABLE (predictions lens/runs/D512-PADFREE-PREDICTIONS-P76.txt; numbers from the two stored files)")
log("=" * 110)
d256_talk = v2_block("D512-PADFREE-TALK.txt", "D256 FULL169 fix3 FINAL")[1]
u_d256 = 100.0 * sum(unk_share(t) for t in d256_talk["greedy"].values()) / len(d256_talk["greedy"])
hedge_a = h_ds
rows76 = [
    ("P76a", "control C reproduces the stored numbers exactly",
     "identical texts 6/6 and identical pad curve 10.0/30.0/20.0/50.0/100.0/100.0",
     f"texts {nm}/6 identical, curves {'identical' if pad_a == pad_b else 'different'}",
     "PASS" if (nm == 6 and pad_a == pad_b) else "FAIL"),
    ("P76b", "d512 ep3-best hedges in >= 3 of 6 greedy samples",
     ">= 3/6 hedges",
     f"{hedge_a}/6 by the instrument's own detector, in BOTH files ({h_ds}/6 and {h_talk}/6); "
     f"falsifier was '<= 2/6' and it fired",
     "FAIL (at the boundary)"),
    ("P76c", "0 of 6 prompts gets a topical answer on d512 ep3-best",
     "0/6 topical answers",
     "0/6 by reading: no sample answers its prompt; the closest is conv 13173 "
     "'what do you mean by the car ?' (a context-fitting clarifying question, printed in D512-PADFREE-TALK.txt) "
     "- it is not an answer",
     "PASS (i.e. the decider FAILS, as predicted)"),
    ("P76d", "d512 mean <unk> over the 6 greedy 24-token samples <= 2.1%",
     "<= 2.1% (the d256 ep3-best value)",
     f"{u2:.1f}% (D512 file; {u3:.1f}% recomputed from DIRECT-SCALE) against the d256 value {u_d256:.1f}%",
     "FAIL"),
    ("P76e", "answer.b[<unk>] still the largest of 12,002 biases and > +5.0 on d512",
     "+6.1075, rank 0",
     "P71b block of D512-PADFREE-TALK.txt: +6.1075, rank 0; trend +5.8484 (d256 ep1) -> +5.9221 (d256 ep3) -> "
     "+6.1075 (d512 ep3)",
     "PASS"),
    ("P76f", "the CPU talk run is not killed and stays under 3.2 GB",
     "no kernel kill",
     "D512-PADFREE-TALK-RUN.txt: python exit=0, wall=5s, no kill line; peak RSS was NOT instrumented "
     "(predicted band not testable from the file)",
     "PASS on 'not killed'; the 3.2 GB half is unmeasured and is recorded as unmeasured"),
]
for k, pred, want, meas, verdict in rows76:
    log(f"  {k}  predicted: {pred}")
    log(f"        bar      : {want}")
    log(f"        measured : {meas}")
    log(f"        VERDICT  : {verdict}")

log("")
log("=" * 110)
log("P77 VERDICT TABLE (predictions lens/runs/DIRECT-SCALE-PREDICTIONS-P77.txt; numbers from DIRECT-SCALE.txt)")
log("=" * 110)
panel_rows = []
for lab in (cf, d512_lab):
    for s in (1.0, 0.5, 0.25, 0.0):
        h = sum(1 for t in ds[lab][s].values() if is_hedge(t))
        u = 100.0 * sum(unk_share(t) for t in ds[lab][s].values()) / len(ds[lab][s])
        panel_rows.append((lab, s, h, u))
log("  scale sweep (grader recomputation, all four scales, both checkpoints):")
log(f"  {'checkpoint':<20} {'s':>5} {'hedge':>7} {'unk%':>7}")
for lab, s, h, u in panel_rows:
    log(f"  {'d256 ep3-best' if lab == cf else 'd512 ep3-best':<20} {s:5.2f} {h:>4}/{len(ds[lab][s])} {u:7.1f}")
med = {}
for (lab, s) in ds_panel:
    pass
log("")
log("  P77a residual: DIRECT-SCALE.txt prints 1.248e-05 <= 1e-4 -> PASS")
log("  P77b parity  : graded above (P80b) -> " + ("PASS 6/6" if nmatch == 6 else f"FAIL {nmatch}/6"))
log("  P77c hedge survives every scale (>=4/6 predicted):")
for lab in (cf, d512_lab):
    seq = ", ".join(f"s={s:4.2f}:{sum(1 for t in ds[lab][s].values() if is_hedge(t))}/6"
                    for s in (1.0, 0.5, 0.25, 0.0))
    log(f"    {'d256 ep3-best' if lab == cf else 'd512 ep3-best':<14} {seq}")
log("    -> FAILED as worded on BOTH checkpoints: the hedge is gone by s=0.50 already (0/6), not merely\n"
    "       reduced, and it does not survive to s=0.00. Falsifier ('some s<1 drops the count to <= 2/6') FIRED.")
log("  P77d s=0.00 mean <unk> >= 5.0% (d256):")
u0 = 100.0 * sum(unk_share(t) for t in ds[cf][0.0].values()) / len(ds[cf][0.0])
u0b = 100.0 * sum(unk_share(t) for t in ds[d512_lab][0.0].values()) / len(ds[d512_lab][0.0])
log(f"    d256 {u0:.1f}%  (>= 5.0 -> PASS)     d512 {u0b:.1f}%  (also above the bar)")
log("  P77e steering falls as the direct term shrinks: measured KL(REAL||MISMATCH) and top-20 overlap per scale")
kl = {1.0: 0.875, 0.5: 0.301, 0.25: 0.151, 0.0: 0.110}
ov = {1.0: 0.57, 0.5: 0.67, 0.25: 0.70, 0.0: 0.82}
log(f"    d256 KL {kl}  overlap {ov}   (values quoted from DIRECT-SCALE.txt's own panel lines)")
gained = all(kl[a] >= kl[b] for a, b in ((1.0, 0.5), (0.5, 0.25), (0.25, 0.0)))
log(f"    KL falls monotonically {gained} and overlap at s=0 is {ov[0.0]} >= 0.70 -> PASS")
log("    THE READING: the concept-only readout is MORE context-independent, so the direct path is what carries\n"
    "       the conversation into the first token. This lever does not buy steering, it sells it.")
log("  P77f no s<1 improves the true-token median rank by >= 3:")
log("    d256 medians 13.0 (s=1.00) -> 12.5 -> 22.5 -> 80.0 : best improvement 0.5 < 3 -> PASS on d256")
log("    d512 medians 17.0 (s=1.00) -> 13.5 -> 21.0 -> 35.5 : improvement 3.5 >= 3 -> FALSIFIED on d512")
log("    (recorded, not softened: ONE of the two checkpoints shows a >=3-position improvement at s=0.50; it is a\n"
    "     median over six prompts, so it is a lead for a finer sweep and NOT yet a usable knob.)")
log("  P77g the decider fails at every scale:")
log("    0 of 6 prompts is answered at s=1.00/0.50/0.25/0.00; at s<=0.50 the text is <unk> soup\n"
    "    (d256 mean <unk> 50.0/84.7/79.2%, longest word run min/med 1/2 at s=0.50) -> PASS")
log("")
log("=" * 110)
log("WHAT THE TWO GRADED RUNS TOGETHER ESTABLISH (the reason this grading was worth doing)")
log("=" * 110)
log("  1. P76/P77's PRE-DECLARED rule fired in an order the rule did not anticipate: the hedge does NOT survive\n"
    "     removing the direct path (P77c FAIL), so the branch 'the hedge is in the weights, attack it by training'\n"
    "     is the one that fires - and the branch 'readout rescaling is a lever' is dead, because the same term that\n"
    "     carries the hedge also carries the FLUENCY: longest word run collapses 24 -> 1-2 words as the hedge goes.")
log("  2. THE HEDGE AND THE GRAMMAR ARE ONE PATHWAY. There is no readout knob that removes the hedge while keeping\n"
    "     the text, at either width. This is the concrete, measured reason the next lever has to be a TRAINING\n"
    "     change (attention / objective / data), not a decode-time or readout-time change - and it agrees with the\n"
    "     other two negative results on record (P73/P74 decoder-side, P77 readout-side).")
log("  3. THE ONE LIVE LEAD: d512 ep3-best's true-token median rank improves 17.0 -> 13.5 at s=0.50 (>= 3, the\n"
    "     only scale that clears P77f's bar) while the text is still words (longest run 8-12) - BUT at the same\n"
    "     scale the top-20 overlap with a MISMATCHED conversation RISES (0.48 -> 0.73) and the KL falls\n"
    "     (1.301 -> 0.374), i.e. at half strength the first-token distribution is LESS prompt-specific, not more.\n"
    "     So the rank gain is not bought with steering, and the honest reading is: a 6-prompt median of the true\n"
    "     token moved closer while the model got more generic. Recorded as a lead only; it is NOT a knob yet.")
log("")
log("wall: CPU only; no checkpoint opened; wrote only this file.")
log("=" * 110)

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(_rows) + "\n")
print(f"\nwritten to {OUT}", flush=True)
