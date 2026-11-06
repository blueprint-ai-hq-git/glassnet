"""gate_cost_numbers.py — THE GATE-COST ABLATION, GRADED FROM STORED TEXT ONLY (predictions P87a-h in
lens/runs/GATE-COST-PREDICTIONS-P87.txt; the grading predictions for THIS instrument are P97a-g in
lens/runs/GATE-COST-GRADE-PREDICTIONS-P97.txt, written before it ran).

WHAT IT DOES: it does not train, does not open a checkpoint, does not use the GPU or a model. It parses
  * lens/runs/GATE-COST.txt                     (the console: epoch lines + exit lines, LAST block only)
  * lens/runs/chat/ours3M_ctrl_std_history.jsonl              (ARM 1, plain SGD, gates OFF)
  * lens/runs/chat/ours3M_relgrad_nogates_history.jsonl       (ARM 2, relgrad step rule, gates OFF)
  * lens/runs/chat/ours_d256_3M_2ep_history.jsonl             (REFERENCE: --method ours, gates ON)
  * lens/runs/PADFREE-TALK-GATECOST.txt         (if present: the SAME instrument run on all three bests)
and prints every graded verdict with the VALUES IT COMPARED printed beside it. Every number is printed with the
file and line it was read from, because this project has three recorded bugs of the class "a right number given a
wrong conclusion" (BUGS.md, verdict-check class).

THE CONTROL IS PART OF THE OUTPUT: the grader's recomputation of each epoch's BY-POSITION column from the jsonl
is printed against the trainer's own console line, so a mismatch is visible as a control failure rather than
silently replacing the record.
"""
import json
import os
import re
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"
CONSOLE = os.path.join(P, "lens/runs/GATE-COST.txt")
ARM1_JSONL = os.path.join(P, "lens/runs/chat/ours3M_ctrl_std_history.jsonl")
ARM2_JSONL = os.path.join(P, "lens/runs/chat/ours3M_relgrad_nogates_history.jsonl")
REF_JSONL = os.path.join(P, "lens/runs/chat/ours_d256_3M_2ep_history.jsonl")
TALK = os.path.join(P, "lens/runs/PADFREE-TALK-GATECOST.txt")
OUT = os.path.join(P, "lens/runs/GRADE-P87.txt")

# P87's own declared constants, quoted from lens/runs/GATE-COST-PREDICTIONS-P87.txt
REF_EP1 = 12.68          # the ours arm's ep1 by-position number, as P87 words it
ARM1_BAND = (25.0, 35.0)
ARM1_FALSIFIER = 20.0
WALL_BAR_MIN = 30.0
RATE_BAR = 2.0           # steps/s
REF_RATE = 1.41          # the reference ours arm's measured per-step rate
REF_MEAN_UNK_STORED = 37.5   # lens/runs/PADFREE-TALK-OURS3M.txt, the instrument's control
REF_MEAN_UNK_TOL = 0.5
GAP_BAR = 5.0            # P87f

EP_RE = re.compile(
    r"^\s+ep\s+(\d+)/(\d+)\s+loss\s+([\d.]+)\s+concept\s+([\d.]+)\s+acc\s+([\d.]+)%\s+"
    r"VAL ppl\s+([\d.]+)\s+acc\s+([\d.]+)%\s+MASKED\(legacy by-id\) ppl\s+([\d.]+)\s+acc\s+([\d.]+)%\s+"
    r"\(n=(\d+)\)\s+ASST\(by-position\) ppl\s+([\d.]+)\s+acc\s+([\d.]+)%\s+\(n=(\d+)\)\s+lr\s+([\d.]+)\s+"
    r"\((\d+)s\)")

_lines = []


def log(s=""):
    _lines.append(str(s))
    print(s, flush=True)


def jl(path):
    """jsonl history -> list of rows, with the file+row printed as the source."""
    rows = []
    with open(path) as f:
        for i, ln in enumerate(f, 1):
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def main():
    log("=" * 100)
    log("THE GATE-COST ABLATION GRADED (P87a-h) - text only, no checkpoint, no model, no GPU")
    log("grading predictions: lens/runs/GATE-COST-GRADE-PREDICTIONS-P97.txt   (written before this instrument)")
    log("the run's own predictions: lens/runs/GATE-COST-PREDICTIONS-P87.txt   (written before the runner was queued)")
    log("=" * 100)

    # ---------- 1. the console's LAST block ----------
    with open(CONSOLE) as f:
        console = f.readlines()
    starts = [i for i, l in enumerate(console) if l.startswith("=== GATE-COST ABLATION")]
    log("")
    log("SOURCE lens/runs/GATE-COST.txt: %d line(s), %d block(s) (the 2026-10-23 21:01 block was killed by that"
        % (len(console), len(starts)))
    log("  night's shutdown at ARM1 exit=143 / ARM2 exit=143 with no epoch line; the LAST block is the graded one).")
    b0 = starts[-1]
    log("  graded block starts at line %d: %s" % (b0 + 1, console[b0].strip()))

    arm_of_line = {}          # line index -> 1 or 2
    cur = 0
    exit_lines = {}
    for i in range(b0, len(console)):
        l = console[i]
        if l.startswith("--- ARM 1:"):
            cur = 1
        elif l.startswith("--- ARM 2:"):
            cur = 2
        elif l.startswith("ARM1 exit=") or l.startswith("ARM2 exit="):
            exit_lines[l.split()[0].replace(" exit=", "")] = (i + 1, l.strip())
        if cur:
            arm_of_line[i] = cur

    con = {1: [], 2: []}
    for i in range(b0, len(console)):
        m = EP_RE.match(console[i])
        if m:
            a = arm_of_line[i]
            con[a].append({"line": i + 1, "epoch": int(m.group(1)),
                           "train_acc": float(m.group(5)), "val_acc": float(m.group(7)),
                           "legacy_acc": float(m.group(9)), "legacy_n": int(m.group(10)),
                           "asst_ppl": float(m.group(11)), "asst_acc": float(m.group(12)),
                           "asst_n": int(m.group(13)), "lr": float(m.group(14)), "seconds": int(m.group(15))})
    for a in (1, 2):
        log("  block ARM %d: %d epoch line(s) parsed  [one per arm is what the recipe asks for at 2 epochs]" % (a, len(con[a])))
    log("  exit lines: ARM1 %s   ARM2 %s" % (exit_lines.get("ARM1", ("?", "MISSING"))[1],
                                             exit_lines.get("ARM2", ("?", "MISSING"))[1]))

    # ---------- 2. the jsonl histories ----------
    h1, h2, hr = jl(ARM1_JSONL), jl(ARM2_JSONL), jl(REF_JSONL)
    log("")
    log("SOURCE %s: %d row(s)" % (ARM1_JSONL.replace(P + "/", ""), len(h1)))
    log("SOURCE %s: %d row(s)" % (ARM2_JSONL.replace(P + "/", ""), len(h2)))
    log("SOURCE %s: %d row(s)" % (REF_JSONL.replace(P + "/", ""), len(hr)))

    # ---------- 3. CONTROL: console vs jsonl, per epoch ----------
    log("")
    log("P97a CONTROL - the grader recomputes BY POSITION from the jsonl and puts it beside the console line:")
    for a, h, name in ((1, h1, "ARM 1 ours3M_ctrl_std"), (2, h2, "ARM 2 ours3M_relgrad_nogates")):
        for row in h:
            e = int(row["epoch"])
            c = [x for x in con[a] if x["epoch"] == e]
            if not c:
                log("  %s ep%d: console line MISSING (jsonl %.4f%%)  -> CONTROL FAIL"
                    % (name, e, 100 * row["val_asst_acc"]))
                continue
            c = c[0]
            d = abs(100 * row["val_asst_acc"] - c["asst_acc"])
            log("  %s ep%d: jsonl %.4f%%  console(line %d) %.2f%%  diff %.4f points  n jsonl %d / console %d  -> %s"
                % (name, e, 100 * row["val_asst_acc"], c["line"], c["asst_acc"], d,
                   row["val_asst_n"], c["asst_n"], "MATCH" if (d <= 0.01 and row["val_asst_n"] == c["asst_n"]) else "MISMATCH"))

    log("")
    log("P97b CONTROL - the reference arm, read from its own file:")
    log("  %s row 1: val_asst_acc %.4f%% = %.4f%%  n=%d  seconds=%.2f"
        % (REF_JSONL.replace(P + "/", ""), 100 * hr[0]["val_asst_acc"], 100 * hr[0]["val_asst_acc"],
           hr[0]["val_asst_n"], hr[0]["seconds"]))
    dref = abs(100 * hr[0]["val_asst_acc"] - REF_EP1)
    log("  P87's own wording of the same number: %.2f%%   difference %.4f points  -> %s"
        % (REF_EP1, dref, "MATCH" if dref <= 0.01 else "MISMATCH"))

    # ---------- 4. the numbers the verdicts compare ----------
    a1_ep1, a1_ep2 = con[1][0], con[1][-1]
    a2_ep1, a2_ep2 = con[2][0], con[2][-1]
    steps = 3000000 // 256
    if steps * 256 != 3000000:
        steps += 1
    ref_ep1 = 100 * hr[0]["val_asst_acc"]
    log("")
    log("THE FOUR NUMBERS EVERY VERDICT BELOW USES (all BY POSITION, n=%d, window cap 3,000,000, batch 256"
        % a1_ep1["asst_n"])
    log("  -> %d steps/epoch; 3,000,000 windows over the whole corpus, seeded 0):" % steps)
    log("    ARM 1 (plain SGD, gates OFF)      ep1 %.2f%%   ep2 %.2f%%  (train %.2f%% / %.2f%%)  wall %d s + %d s"
        % (a1_ep1["asst_acc"], a1_ep2["asst_acc"], a1_ep1["train_acc"], a1_ep2["train_acc"],
           a1_ep1["seconds"], a1_ep2["seconds"]))
    log("    ARM 2 (relgrad, gates OFF)        ep1 %.2f%%   ep2 %.2f%%  (train %.2f%% / %.2f%%)  wall %d s + %d s"
        % (a2_ep1["asst_acc"], a2_ep2["asst_acc"], a2_ep1["train_acc"], a2_ep2["train_acc"],
           a2_ep1["seconds"], a2_ep2["seconds"]))
    log("    REFERENCE (method ours, gates ON) ep1 %.2f%%   (the stored arm, NOT re-run)   wall %.0f s"
        % (ref_ep1, hr[0]["seconds"]))

    log("")
    log("-" * 100)
    log("VERDICTS (each line prints the values it compared, then the verdict; nothing is concluded from memory)")
    log("-" * 100)

    # P87a
    ok = ARM1_BAND[0] <= a1_ep1["asst_acc"] <= ARM1_BAND[1]
    above_ref = a1_ep1["asst_acc"] - ref_ep1
    log("P87a  ARM 1 leaves the reference far behind, ep1 in %s, at least +12 points over %.2f%%;"
        % (list(ARM1_BAND), ref_ep1))
    log("      falsified if ARM 1 is below %.1f%%." % ARM1_FALSIFIER)
    log("      ARM 1 ep1 %.2f%% vs band %s -> inside band: %s ; vs reference: %+.2f points ; vs falsifier %.1f%%: %s"
        % (a1_ep1["asst_acc"], list(ARM1_BAND), ok, above_ref, ARM1_FALSIFIER,
           "falsifier DID fire" if a1_ep1["asst_acc"] < ARM1_FALSIFIER else "falsifier did not fire"))
    log("      -> P87a %s" % ("PASS" if (ok and above_ref >= 12.0) else "FAIL"))

    # P87b
    wall = a1_ep1["seconds"] + a1_ep2["seconds"]
    rate1 = steps / a1_ep1["seconds"]
    rate2 = steps / a2_ep1["seconds"]
    log("")
    log("P87b  ARM 1 two epochs under %.0f min wall and above %.1f steps/s (reference %.2f steps/s):"
        % (WALL_BAR_MIN, RATE_BAR, REF_RATE))
    log("      wall %d s = %.2f min   ARM 1 rate %d/%d = %.2f steps/s (%.1fx the reference)   ARM 2 rate %.2f steps/s"
        % (wall, wall / 60.0, steps, a1_ep1["seconds"], rate1, rate1 / REF_RATE, rate2))
    log("      NOTE: wall here is the sum of the two printed epoch times; the console's own ARM1 exit line gives"
        " the whole arm end to end.")
    log("      -> P87b %s" % ("PASS" if (wall / 60.0 < WALL_BAR_MIN and rate1 > RATE_BAR) else "FAIL"))

    # P87c
    between = ref_ep1 < a2_ep1["asst_acc"] < a1_ep1["asst_acc"]
    margin = a2_ep1["asst_acc"] - ref_ep1
    log("")
    log("P87c  ARM 2 between the reference and ARM 1, and more than 2 points above the reference at matched ep1;")
    log("      falsified if within 2 points of the reference.")
    log("      reference %.2f%% < ARM 2 ep1 %.2f%% < ARM 1 ep1 %.2f%% -> between: %s ; margin over reference %+.2f points"
        % (ref_ep1, a2_ep1["asst_acc"], a1_ep1["asst_acc"], between, margin))
    log("      -> P87c %s" % ("PASS" if (between and margin > 2.0) else "FAIL"))

    # P87d - the attribution, as points
    log("")
    log("P87d  THE ATTRIBUTION, AS POINTS (no verdict, the numbers are the output; gaps below ~1 point would be"
        " inside this project's seed noise and are not acted on):")
    log("      cost of the relgrad STEP RULE  = ARM 1 ep1 - ARM 2 ep1 = %.2f - %.2f = %+.2f points"
        % (a1_ep1["asst_acc"], a2_ep1["asst_acc"], a1_ep1["asst_acc"] - a2_ep1["asst_acc"]))
    log("      cost of the TWO GATES          = ARM 2 ep1 - reference ep1 = %.2f - %.2f = %+.2f points"
        % (a2_ep1["asst_acc"], ref_ep1, a2_ep1["asst_acc"] - ref_ep1))
    log("      cost of the WHOLE METHOD       = ARM 1 ep1 - reference ep1 = %.2f - %.2f = %+.2f points"
        % (a1_ep1["asst_acc"], ref_ep1, a1_ep1["asst_acc"] - ref_ep1))
    log("      the 22.59-point figure this ablation was run to check: reference ep1 %.2f%% against the twin's"
        % ref_ep1)
    log("      ep3 35.263185%% [lens/runs/chat/ct_d256_full169_fix3_history.jsonl] = %+.2f points at MATCHED epoch 3"
        " vs this 3M-window caps's ep1" % (ref_ep1 - 35.263185))
    log("      AT MATCHED SCALE AND MATCHED EPOCH the same comparison on ARM 1 is %.2f - 35.263185 = %+.2f points:"
        % (a1_ep1["asst_acc"], a1_ep1["asst_acc"] - 35.263185))
    log("      the full-corpus twin is a different corpus slice (16.9M vs 3M windows), so this last line is a"
        " REFERENCE POINT, not a like-for-like control.")

    # P87f
    log("")
    log("P87f  gates-off arms show no memorisation signature: train-minus-held-out gap under +%.1f points" % GAP_BAR)
    log("      (by position; falsified above it).")
    for nm, rows in (("ARM 1 ep1", [a1_ep1]), ("ARM 1 ep2", [a1_ep2]), ("ARM 2 ep1", [a2_ep1]), ("ARM 2 ep2", [a2_ep2])):
        r = rows[0]
        log("      %s: train %.2f%% - held-out %.2f%% = %+.2f points -> %s"
            % (nm, r["train_acc"], r["asst_acc"], r["train_acc"] - r["asst_acc"],
               "under the bar" if (r["train_acc"] - r["asst_acc"]) < GAP_BAR else "ABOVE THE BAR"))
    worst = max(x["train_acc"] - x["asst_acc"] for x in (a1_ep1, a1_ep2, a2_ep1, a2_ep2))
    log("      worst gaps-off gap here: %+.2f points -> P87f %s" % (worst, "PASS" if worst < GAP_BAR else "FAIL"))
    log("      for scale, the same column measured elsewhere: twin ep1 +2.25 points"
        " [lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt], tied 12-epoch 1.5M-window run +16.8 points.")

    # ---------- 5. the decider (P87e) and P87g, if the instrument has run ----------
    log("")
    log("-" * 100)
    if not os.path.exists(TALK):
        log("P87e / P87g: NOT MEASURED YET - %s does not exist." % TALK.replace(P + "/", ""))
        log("  The decider is the pad-free talk instrument on all three bests in ONE run"
            " (lens/scripts/run_padfree_gatecost.sh).")
    else:
        log("SOURCE %s" % TALK.replace(P + "/", ""))
        txt = open(TALK).read().splitlines()
        # the v3 SHIP SUMMARY block: label line, then 'mean <unk> share over 6 greedy', then six 'conv' lines
        i0 = None
        for i, l in enumerate(txt):
            if l.startswith("SHIP SUMMARY"):
                i0 = i
        log("  SHIP SUMMARY block at line %s" % (i0 + 1 if i0 is not None else "MISSING"))
        blocks = {}
        if i0 is not None:
            lines_list = txt[i0:]
            cur = None
            for j, l in enumerate(lines_list):
                nxt = lines_list[j + 1] if j + 1 < len(lines_list) else ""
                # A block label is a line that is NOT itself a measurement and whose NEXT line is the
                # 'mean <unk> share over 6 greedy' line. The first version accepted any indented line, so the
                # instrument's own P72e / NOTE lines were read as checkpoint names and printed as three extra
                # 'mean <unk> nan%' blocks (BUGS.md, same turn). The stricter rule cannot do that.
                if (l.startswith("  ") and not l.startswith("      ") and l.strip()
                        and "mean <unk>" not in l and "mean <unk> share over 6 greedy" in nxt):
                    cur = l.strip()
                    blocks.setdefault(cur, {"samples": []})
                    continue
                m = re.search(r"mean <unk> share over 6 greedy 24-token samples:\s+([\d.]+)%\s+"
                              r"samples with a >=4-word run and not all-<unk>:\s+(\d+)/6\s+all-<unk> samples:\s+(\d+)", l)
                if m and cur:
                    blocks[cur]["mean_unk"] = float(m.group(1))
                    blocks[cur]["run_ok"] = int(m.group(2))
                    blocks[cur]["allunk"] = int(m.group(3))
                m2 = re.search(r"^\s+conv\s+(\d+):\s+<unk>\s+([\d.]+)%\s+longest word run\s+(\d+)\s+(.*)$", l)
                if m2 and cur:
                    blocks[cur]["samples"].append({"conv": int(m2.group(1)), "unk": float(m2.group(2)),
                                                   "run": int(m2.group(3)), "text": m2.group(4).strip()})
        for lab, b in blocks.items():
            nsp = sum(1 for s in b["samples"] if s["unk"] >= 33.0 and s["run"] <= 6)
            log("")
            log("  %s" % lab)
            log("    mean <unk> %.1f%%   >=4-word-run %d/6   all-<unk> %d   spiral rule (>=33%% unk and run <=6) fired"
                " on %d/%d sample(s)" % (b.get("mean_unk", float("nan")), b.get("run_ok", 0),
                                         b.get("allunk", 0), nsp, len(b["samples"])))
            for s in b["samples"]:
                log("      conv %5d  <unk> %5.1f%%  run %2d  %s" % (s["conv"], s["unk"], s["run"], s["text"][:110]))

        def mean_for(keyword):
            for lab, b in blocks.items():
                if keyword.lower() in lab.lower():
                    return lab, b.get("mean_unk")
            return None, None

        lab1, m1 = mean_for("ARM1")
        lab2, m2 = mean_for("ARM2")
        labr, mr = mean_for("REFERENCE")
        log("")
        log("-" * 100)
        log("P87g  <unk> SHARE FOLLOWS TRAINING, NOT THE BIAS TERM: ARM 1 at or below 20% (falsified above 30%),"
            " against the reference arm's 37.5% measured by the same instrument.")
        log("      ARM 1  mean <unk> %.1f%%   (%s)" % (m1 if m1 is not None else float("nan"), lab1))
        log("      ARM 2  mean <unk> %.1f%%   (%s)" % (m2 if m2 is not None else float("nan"), lab2))
        log("      REF    mean <unk> %.1f%%   (%s)" % (mr if mr is not None else float("nan"), labr))
        if m1 is None:
            log("      -> CANNOT GRADE: ARM 1 block not found in the summary.")
        elif m1 <= 20.0:
            log("      %.1f%% <= 20.0%% (and %s 30.0%% falsifier) -> P87g PASS" % (m1, "<" if m1 < 30 else ">"))
        elif m1 <= 30.0:
            log("      %.1f%% is ABOVE the 20.0%% bar (falsifier at 30.0%% did NOT fire) -> P87g MISSED MY OWN BAND,"
                " not falsified as worded" % m1)
        else:
            log("      %.1f%% is above the pre-declared 30.0%% falsifier -> P87g FAIL" % m1)
        log("      P97d CONTROL on the instrument: the reference arm in THIS run must read 37.5%"
            " [lens/runs/PADFREE-TALK-OURS3M.txt].")
        if mr is not None:
            d = abs(mr - REF_MEAN_UNK_STORED)
            log("      measured here %.1f%% vs stored %.1f%%  diff %.1f points -> %s"
                % (mr, REF_MEAN_UNK_STORED, d, "MATCH (control holds)" if d <= REF_MEAN_UNK_TOL else "CONTROL FAILED"))
            log("      MY OWN P97e EXPECTATION (ARM 1 in 5-20%%): %s"
                % ("held" if (m1 is not None and 5.0 <= m1 <= 20.0) else "MISSED - recorded, not smoothed"))
        log("")
        log("P87e  THE DECIDER - all three arms still fail coherence; falsified if ANY shows 2 or more coherent")
        log("      answers. THE VERDICT IS BY EYE ON THE SAMPLES ABOVE (that is how this project counts"
            " coherence); the spiral rule and the counts are consistency evidence only. Read the six samples per")
        log("      arm against their prompts in the TALK section of the same file, which prints prompt + TRUE REPLY.")

    log("")
    log("=" * 100)
    log("NOTHING ABOVE IS A QUALITY CLAIM ABOUT ANY ARM BEYOND THE COLUMNS NAMED. Every accuracy is by position")
    log("(assistant turns only, n=58,191) unless the line says LEGACY by token id, which is a share over a")
    log("different, polluted target set (BUG-MASK-TOKEN-ID) and is printed only for comparison.")
    log("=" * 100)

    with open(OUT, "w") as f:
        f.write("\n".join(_lines) + "\n")
    print("\n[written] %s (%d lines)" % (OUT, len(_lines)))


if __name__ == "__main__":
    main()
