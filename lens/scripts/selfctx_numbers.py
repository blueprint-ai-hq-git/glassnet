#!/usr/bin/env python3
"""SELF-CONTEXT ARM — THE NUMBERS, AND THE VERDICTS BESIDE THEM.  Text only: no checkpoint is opened, no model
is built, no GPU is used, no trainer is started.

Predictions: lens/runs/SELFCTX-EARLY-PREDICTIONS-P99.txt (written BEFORE this file existed). P99a is the control:
every row printed here is recomputed from the two stored jsonl files and printed BESIDE the value it claims to
reproduce, so a wrong number cannot hide and a verdict can never be printed without the values it came from.

Sources (read-only):
  arm  : lens/runs/chat/sctx_d256_full169_history.jsonl   (the self-context arm, --self-context-frac 0.5 rounds 4)
  twin : lens/runs/chat/ct_d256_full169_fix3_history.jsonl (the MATCHED control: same corpus, d, window, batch,
         lr, seed, tied path, assistant-only by position, train-turn-end; the ONE difference is the two flags)
  console: lens/runs/SELF-CONTEXT-TIED.txt
"""
import json
import os
import re
import sys

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../GlassNet
ARM_JSONL = os.path.join(P, "lens/runs/chat/sctx_d256_full169_history.jsonl")
TWIN_JSONL = os.path.join(P, "lens/runs/chat/ct_d256_full169_fix3_history.jsonl")
LOG = os.path.join(P, "lens/runs/SELF-CONTEXT-TIED.txt")
EARLY = os.path.join(P, "lens/runs/PADFREE-TALK-SELFCTX-EP1.txt")

P98D_BAND = 3.0          # P98d's pre-declared band, in points, against the twin at the same epoch


def rows(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def pc(x):
    return "%.4f%%" % (100.0 * x)


def main():
    arm = rows(ARM_JSONL)
    twin = rows(TWIN_JSONL)
    print("=" * 100)
    print("SELF-CONTEXT / EXPOSURE-BIAS ARM — THE NUMBERS AND THEIR VERDICTS (P99a-e)")
    print("predictions: lens/runs/SELFCTX-EARLY-PREDICTIONS-P99.txt  (written before this script existed)")
    rel = lambda p: os.path.relpath(p, P)
    print("sources: arm  lens/runs/chat/sctx_d256_full169_history.jsonl  -> %s  (%d row(s))"
          % ("PRESENT" if os.path.exists(ARM_JSONL) else "MISSING", len(arm)))
    print("         twin lens/runs/chat/ct_d256_full169_fix3_history.jsonl -> %s  (%d row(s))"
          % ("PRESENT" if os.path.exists(TWIN_JSONL) else "MISSING", len(twin)))
    print("=" * 100)
    if not arm or not twin:
        print("\nNO VERDICT IS POSSIBLE: %d arm row(s), %d twin row(s) read from disk. This script prints NO")
        print("share, NO delta and NO band verdict from an empty input — an empty read is a FAILED CONTROL,")
        print("not a clean control. Name the missing file above before quoting anything from this run.")
        return 1

    # ---------------- P99a CONTROL: print the reproduced value beside the field it came from ----------------
    print("\nP99a CONTROL — every row recomputed from the jsonl field it claims, printed beside it:")
    ok = True
    for label, rs in (("arm ", arm), ("twin", twin)):
        for r in rs:
            v = r.get("val_asst_acc")
            n = r.get("val_asst_n")
            t = r.get("train_acc")
            print("  %s ep%-2d  val_asst_acc field=%.10f  printed=%s  (n=%s)   train_acc field=%.10f printed=%s"
                  % (label, r.get("epoch"), v, pc(v), n, t, pc(t)))
            if not (isinstance(v, float) and 0.0 <= v <= 1.0 and n == 58191):
                ok = False
                print("      !! FIELD OUT OF RANGE OR WRONG n — this row is NOT usable as a number")
    print("  CONTROL VERDICT: %s  (%d arm row(s) + %d twin row(s), all printed with their primary field beside "
          "the verdict share)" % ("PASS" if ok else "FAIL — the jsonl is not what this script expects",
                                  len(arm), len(twin)))
    if not arm:
        print("\nNO ARM ROWS YET: the arm has written no epoch line. Nothing below can be a verdict.")
        return 0 if ok else 1

    # ---------------- P98d: the band, per epoch, both values printed ----------------
    print("\n" + "-" * 100)
    print("P98d — HELD-OUT BY POSITION AGAINST THE MATCHED TWIN AT THE SAME EPOCH (pre-declared band +-%.1f "
          "points)" % P98D_BAND)
    print("  ep | arm by-position (n)          | twin by-position (n)          | delta (arm - twin) | P98d band")
    ep3_band_pass = False
    for r in arm:
        e = r.get("epoch")
        a = r.get("val_asst_acc")
        t = next((x.get("val_asst_acc") for x in twin if x.get("epoch") == e), None)
        if t is None:
            print("  %2d | %-27s | (twin has no row for this epoch)" % (e, pc(a)))
            continue
        d = 100.0 * (a - t)
        inband = abs(d) <= P98D_BAND
        if e == max(x.get("epoch") for x in arm) and len(arm) == 3:
            ep3_band_pass = inband
        print("  %2d | %-27s | %-27s | %+7.2f points      | %s"
              % (e, "%s (n=%s)" % (pc(a), r.get("val_asst_n")), "%s (n=%s)" % (pc(t), 58191), d,
                 "inside" if inband else "OUTSIDE -> FAIL"))
    print("  P99c's own prediction: the arm's final best lands in 27.0-30.0% (a price of 5-8 points).")
    if len(arm) == 3:
        best = max(r.get("val_asst_acc") for r in arm)
        print("  P99c band 27.0-30.0%%: best so far %s -> %s" % (pc(best),
              "inside" if 0.27 <= best <= 0.30 else "OUTSIDE"))
        print("  P98d band +-3 points AT THE FINAL EPOCH: %s" % ("PASS" if ep3_band_pass else "FAIL"))
    else:
        print("  P98d's FINAL-EPOCH verdict is UNMEASURED: %d of 3 epochs have a row so far "
              "(%s). Only the epochs printed above are graded; none of the rest is estimated."
              % (len(arm), ", ".join("ep%d" % r.get("epoch") for r in arm)))

    # ---------------- train column + the memorisation question (plan item 3) ----------------
    print("\n" + "-" * 100)
    print("TRAIN vs HELD-OUT, WITHIN THE ARM ONLY (the two columns are shares over DIFFERENT target sets between "
          "arms: train_acc is over ALL positions, val_asst_acc over assistant positions, so a cross-arm "
          "subtraction of this gap would be a lie — it is printed as a WITHIN-ARM trend)")
    print("  ep | arm train | arm held-out | arm gap (train - held-out) | twin gap")
    for r in arm:
        e = r.get("epoch")
        g = 100.0 * (r["train_acc"] - r["val_asst_acc"])
        tr = next((x for x in twin if x.get("epoch") == e), None)
        gt = "n/a" if tr is None else "%+6.2f points" % (100.0 * (tr["train_acc"] - tr["val_asst_acc"]))
        print("  %2d | %9s | %12s | %+26.2f | %s" % (e, pc(r["train_acc"]), pc(r["val_asst_acc"]), g, gt))
    if len(arm) >= 2:
        a0, a1 = arm[-2], arm[-1]
        print("  last two epochs: held-out %s -> %s (%s), train %s -> %s (%s)"
              % (pc(a0["val_asst_acc"]), pc(a1["val_asst_acc"]),
                 "up" if a1["val_asst_acc"] > a0["val_asst_acc"] else "DOWN",
                 pc(a0["train_acc"]), pc(a1["train_acc"]),
                 "up" if a1["train_acc"] > a0["train_acc"] else "DOWN"))

    # ---------------- cost (P99d / P98g) ----------------
    print("\n" + "-" * 100)
    print("COST (P99d / P98g):")
    tot = sum(r.get("seconds", 0.0) for r in arm)
    for r in arm:
        t = next((x.get("seconds") for x in twin if x.get("epoch") == r.get("epoch")), None)
        ratio = "" if not t else "  (%.2fx the twin's %.1f s)" % (r["seconds"] / t, t)
        print("  ep%d  %8.1f s%s" % (r["epoch"], r["seconds"], ratio))
    print("  epochs so far total %.1f s = %.2f h for %d epoch(s); P98g predicted 1.5-2.5 h for all three and "
          "falsified above 4 h; P99d predicted 1.2-2.2 h" % (tot, tot / 3600.0, len(arm)))

    # ---------------- P99b: the EARLY talk read, parsed from the instrument's own file ----------------
    print("\n" + "-" * 100)
    print("P99b — THE EARLY TALK READ (arm's EPOCH-1 BEST vs the stored TWIN, ONE instrument run):")
    if os.path.exists(EARLY):
        blocks = []
        with open(EARLY) as fh:
            in_sum = False
            cur = None
            for line in fh:
                if "SHIP SUMMARY" in line:
                    in_sum = True
                    continue
                if not in_sum:
                    continue
                line = line.rstrip("\n")
                m = re.search(r"mean <unk> share over 6 greedy 24-token samples:\s*([0-9.]+)%\s*"
                              r"samples with a >=4-word run and not all-<unk>:\s*(\d+)/(\d+)\s*all-<unk> samples:\s*(\d+)",
                              line)
                if m:
                    if cur is not None:
                        blk = dict(cur)
                        blk.update(mean_unk=float(m.group(1)), runs=m.group(2), n=m.group(3),
                                   allunk=m.group(4))
                        blocks.append(blk)
                    continue
                m2 = re.match(r"^\s{2}(\S.*)$", line)
                if m2 and not line.startswith("      ") and not line.startswith("===="):
                    cur = {"label": m2.group(1).strip()}
        for b in blocks:
            print("  %-95s" % b["label"][:95])
            print("      mean <unk> over 6 greedy 24-token samples: %5.1f%%   >=(4)-word run: %s/%s   all-<unk>: %s"
                  % (b["mean_unk"], b["runs"], b["n"], b["allunk"]))
        if len(blocks) >= 2:
            arm_b, twin_b = blocks[0], blocks[-1]
            print("  CONTROL (the twin, measured in the SAME run) must reproduce its stored 2.1%% and 6/6: "
                  "read %.1f%% and %s/%s -> %s"
                  % (twin_b["mean_unk"], twin_b["runs"], twin_b["n"],
                     "PASS" if abs(twin_b["mean_unk"] - 2.1) < 0.05 and twin_b["runs"] == twin_b["n"] else "FAIL"))
            print("  P99b(ii) my bar was UNDER 5%% on the arm: read **%.1f%%** against the twin's %.1f%% -> %s"
                  % (arm_b["mean_unk"], twin_b["mean_unk"],
                     "HELD" if arm_b["mean_unk"] < 5.0 else "FALSIFIED (the arm FLOODS <unk> worse than the twin)"))
            print("  P99b(i) 0/6 prompt-relevant answers, BY EYE on the six prompt/true-reply/greedy blocks of %s"
                  % os.path.relpath(EARLY, P))
            print("  P99b(iii) the twin's column above is the instrument control, so the arm's column is readable.")
        else:
            print("  COULD NOT PARSE the instrument's SHIP SUMMARY (%d block(s) found) — no P99b verdict is "
                  "printed from it." % len(blocks))
    else:
        print("  UNMEASURED: %s does not exist." % os.path.relpath(EARLY, P))

    # ---------------- what is NOT measured yet, and from where it will come ----------------
    print("\n" + "-" * 100)
    print("STILL UNMEASURED IN THIS FILE, NAMED WITH THE FILE IT WILL COME FROM (no estimate is offered):")
    print("  * P98e the lever's own falsifier (step-8 free-running above the twin's 3.9%): chained in the live "
          "job, will be lens/runs/TALK-FAIR-SELFCTX.txt with lens/runs/TALK-FAIR-SELFCTX-TWIN.txt as the "
          "instrument's control.")
    print("  * P98f the ship decider (2+ prompt-relevant answers): chained in the live job, will be "
          "lens/runs/PADFREE-TALK-SELFCTX.txt, read by eye.")
    print("  * P99b the EARLY talk read on the epoch-1 best: lens/runs/PADFREE-TALK-SELFCTX-EP1.txt (separate "
          "instrument run, same day, arm's ep1-best + the twin in one run).")
    print("  * the arm's prefix as a whole: does it finish (ARM EXIT=) — lens/runs/SELF-CONTEXT-TIED.txt.")
    print("\nNOTHING IS SHIPPED FROM THIS ARM ON THESE NUMBERS. A held-out column is not the ship criterion; "
          "P98f is.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
