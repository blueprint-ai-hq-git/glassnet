#!/usr/bin/env python3
"""grade_unk_drop_preflight.py — the CORRECTED P89a-c verdicts for the --drop-unk pre-flight, from the stored
text only. Written 2026-10-24 08:3x, AFTER the pre-flight ran and AFTER ADDENDUM 2/3 of
lens/runs/UNK-DROP-PREDICTIONS-P89.txt were written, and BEFORE the full-corpus arm exists.

WHY IT EXISTS: the pre-flight's own inline grading printed P89c as DISAGREE because its criterion compared the
trainer's count over the BUILT window set against an independent count over the CAPPED one (300,000 vs 60,000).
The criterion was fixed in ADDENDUM 2 and the cause was measured in ADDENDUM 3 (the drop-unk block was moved
below the cap). This grader applies the CORRECTED criterion and keeps the original DISAGREE visible in
lens/runs/UNK-DROP-PREFLIGHT.txt rather than rewriting it.
No model, no GPU, no checkpoint: it reads three text files and prints every number with its file.
"""
import os
import re
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"
LOG = os.path.join(P, "lens/runs/UNK-DROP-PREFLIGHT.txt")
TMP = os.path.join(P, "lens/runs/_tmp")
OUT = os.path.join(P, "lens/runs/GRADE-PREFLIGHT-P89.txt")
# the stored P71f figure, quoted with the files that carry it
P71F = [("lens/runs/PADFREE-TALK.txt", "line 4"),
        ("lens/runs/ATTN-GRADE-RUN.txt", "line 211"),
        ("lens/runs/D512-PADFREE-TALK.txt", "line 4")]


class Tee:
    def __init__(self, path):
        self.f = open(path, "a")
    def w(self, s=""):
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()
    def close(self):
        self.f.close()


def epoch_line(txt):
    for ln in txt.splitlines():
        if re.match(r"\s*ep\s+\d+/\d+", ln) and "loss" in ln:
            return ln.strip()
    return None


def strip_time(s):
    return re.sub(r"\s*\(\d+s\)\s*$", "", s) if s else s


def read_arm(tag):
    p = os.path.join(TMP, f"unkpf_{tag}.out")
    return open(p).read() if os.path.exists(p) else None


def main():
    T = Tee(OUT)
    T.w()
    T.w("=" * 94)
    T.w("CORRECTED P89a-c GRADING — the --drop-unk CPU pre-flight (d=64, WIRING ONLY, no quality claim)")
    T.w("predictions: lens/runs/UNK-DROP-PREDICTIONS-P89.txt (P89a-c; criterion re-worded in ADDENDUM 2, cause")
    T.w("             measured in ADDENDUM 3). The pre-flight's own first grading is left as it was written; it")
    T.w("             is read below so its DISAGREE stays on the record instead of being silently replaced.")
    T.w("=" * 94)
    T.w()

    arms = {t: read_arm(t) for t in ("prechange", "newdefault", "dropunk")}
    for t, v in arms.items():
        T.w(f"  {'FOUND   ' if v is not None else 'MISSING '} lens/runs/_tmp/unkpf_{t}.out"
            + (f"   ({len(v):,} chars)" if v else ""))
    if not all(arms.values()):
        T.w("ONE OF THE THREE ARMS IS MISSING — no verdict is issued from an incomplete pre-flight.")
        T.close()
        return
    T.w()

    ep_pre = epoch_line(arms["prechange"])
    ep_new = epoch_line(arms["newdefault"])
    ep_dr = epoch_line(arms["dropunk"])

    T.w("-" * 94)
    T.w("P89a — THE DEFAULT PATH IS UNCHANGED (new trainer, --drop-unk ABSENT, against the pre-change copy)")
    T.w("-" * 94)
    T.w(f"  pre-change: {ep_pre}")
    T.w(f"  new file  : {ep_new}")
    full = (ep_pre is not None and ep_pre == ep_new)
    res = (ep_pre is not None and strip_time(ep_pre) == strip_time(ep_new))
    newdrop = [l for l in arms["newdefault"].splitlines() if "drop-unk" in l]
    T.w(f"  identical as a whole line (timing field included): {full}")
    T.w(f"  identical after stripping the (NNNs) wall-clock field, which is NOT a measurement of the model: {res}")
    T.w(f"  lines mentioning drop-unk in the DEFAULT run: {len(newdrop)} (predicted 0)")
    T.w(f"  -> P89a {'PASS' if (res and not newdrop) else 'FAIL'}")
    T.w()

    T.w("-" * 94)
    T.w("P89b — THE FLAG IS WIRED AND NOT A NO-OP")
    T.w("-" * 94)
    # THE ARM GRADED HERE IS THE CODE AS SHIPPED. The pre-flight's dropunk arm ran with the block ABOVE the cap;
    # the block was then moved below it (ADDENDUM 3) and re-run as unkpf_movecheck, which is the current code.
    # Both print-outs are quoted so the change is visible rather than implied.
    cur = read_arm("movecheck")
    if cur is None:
        T.w("  unkpf_movecheck.out (the post-move run) is MISSING: grading the pre-move arm and SAYING SO.")
        cur = arms["dropunk"]
    T.w("  printed by the pre-move block (kept for the record):")
    for l in [x.strip() for x in arms["dropunk"].splitlines() if "drop-unk:" in x]:
        T.w("    " + l)
    T.w("  printed by the block as shipped (below the window cap):")
    for l in [x.strip() for x in cur.splitlines() if "drop-unk:" in x]:
        T.w("    " + l)
    d_all = d_cap = d_w = None
    for ln in cur.splitlines():
        if "drop-unk:" in ln and "removed from the loss" in ln:
            m = re.search(r"drop-unk: ([\d,]+) training targets", ln)
            c = re.search(r"([\d.]+)% of the ([\d,]+) training targets this run actually trains on", ln)
            old = re.search(r"([\d.]+)% of all training targets", ln)
            w = re.search(r"([\d.]+)% of the WEIGHTED", ln)
            d_all = int(m.group(1).replace(",", "")) if m else None
            if c:
                d_cap = (float(c.group(1)), int(c.group(2).replace(",", "")))
            elif old:
                d_cap = (float(old.group(1)), None)      # pre-move wording: denominator is the BUILT set
            d_w = float(w.group(1)) if w else None
    exits = [int(m.group(1)) for m in re.finditer(r"^(?:ARM |trainer )?exit=(\d+)", cur, re.M)]
    ex_src = "lens/runs/_tmp/unkpf_movecheck.out"
    if not exits:
        # THE MOVE-CHECK ARM'S EXIT LINE IS NOT IN THE ARM FILE: unk_drop_movecheck.sh writes it to its own log
        # (the arm file only holds the trainer's stdout). Read it where it is and SAY which file it came from —
        # otherwise a missing line reads as "the run did not exit cleanly", which is the opposite of the truth.
        mlog = os.path.join(P, "lens/runs/UNK-DROP-MOVECHECK.txt")
        if os.path.exists(mlog):
            exits = [int(m.group(1)) for m in re.finditer(r"^trainer exit=(\d+)", open(mlog).read(), re.M)]
            ex_src = "lens/runs/UNK-DROP-MOVECHECK.txt (trainer exit line; the arm file holds only stdout)"
    # EVERY RECORDED EXIT FOR THIS ARM MUST BE 0 (the move-check log can hold more than one run: the first
    # version of this criterion demanded exactly `[0]`, which fails on `[0, 0]` — two clean runs — a verdict
    # read off the SHAPE of the list rather than its values).
    clean = bool(exits) and all(e == 0 for e in exits)
    ep_cur = epoch_line(cur)
    if d_cap:
        denom = (f"{d_cap[1]:,} training targets this run trains on" if d_cap[1]
                 else "ALL training targets (pre-move wording: the BUILT set, not the trained one)")
        T.w(f"  dropped {d_all:,} targets = {d_cap[0]}% of the {denom}"
            f" ({d_w}% of the weighted assistant ones); exits {exits}   [{ex_src}]")
    else:
        T.w("  COULD NOT PARSE THE DROP LINE — no verdict is issued from it. Lines seen: "
            + str([x for x in cur.splitlines() if "drop-unk:" in x]))
    T.w(f"  the drop-unk arm's epoch line differs from the default arm's: {ep_cur != ep_new}")
    ok_b = (d_all is not None and d_all > 0 and clean and ep_cur != ep_new)
    T.w(f"  -> P89b {'PASS' if ok_b else 'FAIL'}   (non-zero count, clean exit, and the run is not bit-identical"
        f" to the flagless one)")
    T.w()

    T.w("-" * 94)
    T.w("P89c — CORRECTED CRITERION (ADDENDUM 2): (i) the trainer's assistant-<unk> share must be <= the")
    T.w("      independent ALL-TARGET <unk> share over the same window set; (ii) the held-out line must")
    T.w("      reproduce the stored P71f figure exactly; (iii) the band 4-9% of WEIGHTED targets dropped.")
    T.w("-" * 94)
    rep = re.search(r"independent all-target <unk> share\s*:\s*([\d.]+)%", open(LOG).read())
    indep = float(rep.group(1)) if rep else None
    tr_all = None
    m = re.search(r"drop-unk: ([\d,]+) training targets .*?-- ([\d.]+)% of (?:all|the [\d,]+) training targets",
                  cur, re.S)
    if m:
        tr_all = float(m.group(2))
    T.w(f"  independent ALL-TARGET <unk> share, rebuilt from the data by the pre-flight: {indep}%"
        f"   [{LOG}, its P89c block]")
    T.w(f"  trainer's assistant-<unk> share of the training targets: {tr_all}%   "
        f"[lens/runs/_tmp/unkpf_movecheck.out]" if cur is not None else "")
    T.w(f"  (the pre-flight's own first grading compared these two and printed DISAGREE; they are different"
        f" sets, and ADDENDUM 3 measured why: the trainer's number was taken over the BUILT 300,000-window set"
        f" while the replica's was over the CAPPED 60,000; the block now runs below the cap)")
    sub_ok = (indep is not None and tr_all is not None and tr_all <= indep)
    T.w(f"  (i) subset relation (assistant-only <= all-target): {tr_all}% <= {indep}% -> "
        f"{'HOLDS' if sub_ok else 'VIOLATED'}")
    unkva = re.search(r"held-out assistant targets that are <unk>: ([\d,]+) of ([\d,]+) \(([\d.]+)%\)",
                      cur)
    if unkva:
        got = float(unkva.group(3))
        T.w(f"  (ii) held-out assistant <unk>: {unkva.group(1)} of {unkva.group(2)} = {got}%  against the stored")
        for f, ln in P71F:
            T.w(f"       P71f 6.888%  [{f} {ln}]")
        T.w(f"       -> {'REPRODUCES EXACTLY' if abs(got - 6.888) < 0.0005 else 'DOES NOT REPRODUCE'}")
    T.w(f"  (iii) band 4-9% of WEIGHTED dropped: measured {d_w}% -> "
        f"{'INSIDE' if (d_w is not None and 4.0 <= d_w <= 9.0) else 'OUTSIDE'}")
    ok_c = sub_ok and unkva is not None and abs(float(unkva.group(3)) - 6.888) < 0.0005 and 4.0 <= d_w <= 9.0
    T.w(f"  -> P89c {'PASS' if ok_c else 'FAIL'}")
    T.w()
    T.w("WIRING ONLY. Nothing about quality is claimed from d=64 numbers, and nothing is shipped from any of this")
    T.w("(P89g). The arm itself is queued as 'bash lens/scripts/run_unk_drop.sh'.")
    T.w("=" * 94)
    T.close()


if __name__ == "__main__":
    sys.exit(main())
