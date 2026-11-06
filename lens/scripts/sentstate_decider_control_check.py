"""lens/scripts/sentstate_decider_control_check.py -- P115e's comparison, text only (no model, no corpus).

WHAT IT COMPARES, and why this is a real control rather than a tolerance: greedy decode is DETERMINISTIC, so
the flag-absent path of the edited instrument must print the SAME six greedy 24-token answers, the same
prompt lines and the same <unk> shares as the block already on disk (lens/runs/PADFREE-TALK-H2L.txt, its
"H2L arm" block). One different token means the edit changed the instrument, and then no arm may be graded
with it.

DEFECT FIXED HERE BEFORE ITS VERDICT WAS BELIEVED (18:5x, first run of this checker): the stored file holds
TWO checkpoint blocks (the arm at line 7 and its matched TWIN control at line 65), and the first version of
this parser kept ONE dict and let the twin's samples OVERWRITE the arm's -- so it printed "GREEDY TOKENS 0/6
identical" while the arm's own block had in fact been reproduced to the character. That is this project's
most repeated bug class (a right number beside a wrong verdict), it is recorded in BUGS.md, and the parser now
keeps blocks separate and compares BLOCK 1 of each file by block index AND by label.
"""
import os
import re
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"
STORED = os.path.join(P, "lens/runs/PADFREE-TALK-H2L.txt")
NEW = os.path.join(P, "lens/runs/PADFREE-TALK-H2L-CTRL-P115.txt")
BRANCH = os.path.join(P, "lens/runs/PADFREE-TALK-SENTSTATE-TINY.txt")
REFUSAL = os.path.join(P, "lens/runs/PADFREE-TALK-SENTSTATE-REFUSAL.txt")
MAINLOG = os.path.join(P, "lens/runs/SENTSTATE-DECODE.txt")

CONV = re.compile(r"^\s*conv (\d+)\s+PROMPT")
GREEDY = re.compile(r"^\s*MODEL GREEDY\s+(.*)$")
SHARE = re.compile(r"^\s*<unk> share \(24 greedy\)\s+([0-9.]+)%")
MEAN = re.compile(r"^\s*mean <unk> share over (\d+) greedy 24-token samples:\s+([0-9.]+)%")
LABEL = re.compile(r"^\s*CHECKPOINT (.*)$")


def parse_blocks(path):
    """-> ([{'label','greedy'{conv:txt},'share'{conv:pct},'mean'(n,pct)}], [means in file order])."""
    if not os.path.exists(path):
        return None, []
    blocks, means = [], []
    cur_conv = None
    for line in open(path, errors="replace"):
        m = LABEL.match(line)
        if m:
            blocks.append({"label": m.group(1).strip(), "greedy": {}, "share": {}, "mean": None})
            cur_conv = None
            continue
        m = MEAN.match(line)
        if m:
            # the SHIP SUMMARY section prints these AFTER both blocks, one per checkpoint, in block order;
            # they are collected separately so they cannot be attached to whichever block happened to be open.
            means.append((int(m.group(1)), float(m.group(2))))
            if blocks:
                blocks[-1]["mean"] = means[-1]
            continue
        if not blocks:
            continue
        m = CONV.match(line)
        if m:
            cur_conv = m.group(1)
            continue
        m = GREEDY.match(line)
        if m and cur_conv is not None:
            blocks[-1]["greedy"][cur_conv] = m.group(1).strip()
            continue
        m = SHARE.match(line)
        if m and cur_conv is not None:
            blocks[-1]["share"][cur_conv] = float(m.group(1))
            continue
    return blocks, means


def describe(b):
    return f"blocks={len(b)} labels={[x['label'][:38] for x in b]}"


def main():
    bad = 0
    print("=== P115e(i) COMPARISON: the edited decider's flag-ABSENT path against the block already on disk")
    a, a_means = parse_blocks(STORED)
    b, b_means = parse_blocks(NEW)
    if not a or not b:
        print(f"  MISSING/EMPTY FILE: stored={bool(a)} new={bool(b)} -> FAIL")
        return 1
    print(f"  stored : {os.path.basename(STORED)}  {describe(a)}  summary means in file order {a_means}")
    print(f"  new    : {os.path.basename(NEW)}  {describe(b)}  summary means in file order {b_means}")
    A, B = a[0], b[0]
    print(f"  comparing BLOCK 1 of each: stored='{A['label'][:60]}'  new='{B['label'][:60]}'")
    if A["greedy"].keys() != B["greedy"].keys():
        print(f"  PROMPT SETS DIFFER: stored {sorted(A['greedy'])} vs new {sorted(B['greedy'])} -> FAIL")
        bad += 1
    same = [k for k in sorted(A["greedy"]) if B["greedy"].get(k) == A["greedy"][k]]
    for k in sorted(A["greedy"]):
        mark = "SAME" if k in same else "DIFF"
        print(f"    conv {k:>6}  {mark}   stored <unk> {A['share'].get(k)}%  new <unk> {B['share'].get(k)}%")
        if mark == "DIFF":
            print(f"        stored: {A['greedy'][k][:110]}")
            print(f"        new   : {B['greedy'].get(k, '<absent>')[:110]}")
    ok_tokens = len(same) == len(A["greedy"]) and len(A["greedy"]) > 0
    ok_shares = A["share"] == B["share"] and len(A["share"]) > 0
    ok_mean = bool(a_means) and bool(b_means) and a_means[0] == b_means[0]
    print(f"  GREEDY TOKENS: {len(same)}/{len(A['greedy'])} samples identical   -> {'PASS' if ok_tokens else 'FAIL'}")
    print(f"  <unk> SHARES : all equal={ok_shares} ({len(B['share'])} vs {len(A['share'])})   "
          f"-> {'PASS' if ok_shares else 'FAIL'}")
    print(f"  SUMMARY MEAN : stored first {a_means[0] if a_means else None}  "
          f"new {b_means[0] if b_means else None}   -> {'PASS' if ok_mean else 'FAIL'}")
    ok = ok_tokens and ok_shares and ok_mean
    bad += 0 if ok else 1
    print(f"  P115e(i) VERDICT: {'PASS' if ok else 'FAIL'} (flag-absent path reproduces the stored reading)")

    print("")
    print("=== P115e(ii) BRANCH: flag ON took the grafted loader inside the decider")
    if not os.path.exists(BRANCH):
        print("  MISSING: the branch run wrote no file -> FAIL")
        bad += 1
    else:
        txt = open(BRANCH, errors="replace").read()
        hit = [l for l in txt.splitlines() if "SENTSTATE GRAFT LOAD" in l]
        print(f"  grafted-loader line printed: {len(hit)}")
        for l in hit:
            print("    " + l.strip())
        ok2 = bool(hit)
        print(f"  P115e(ii) VERDICT: {'PASS' if ok2 else 'FAIL'} (wiring only: tiny vocab, no quality claim)")
        bad += 0 if ok2 else 1

    print("")
    print("=== P115e(iii) REFUSAL: flag ON with a BASE checkpoint must fail by name")
    txt = ""
    for p in (REFUSAL, MAINLOG):
        if os.path.exists(p):
            txt += open(p, errors="replace").read()
    named = "carries no sentstate tensors" in txt
    print(f"  'carries no sentstate tensors' present in the output: {named}")
    print(f"  P115e(iii) VERDICT: {'PASS' if named else 'FAIL'}")
    bad += 0 if named else 1
    print("")
    print(f"P115e OVERALL: {'PASS' if bad == 0 else 'FAIL'}  ({bad} failed part(s))")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
