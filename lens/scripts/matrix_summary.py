#!/usr/bin/env python3
"""MATRIX SUMMARY — rank the comparison-matrix runs on the common yardstick and name the winner.

Reads lens/runs/MATRIX.txt (written by run_matrix.sh) and parses, per variant:
  * (A) the MERGED clean-val assistant-masked accuracy/ppl  <- the intended common yardstick
  * (B) the variant's own-corpus clean-val accuracy
It ranks the variants and names the winner with its margin.

VOCABULARY-FAMILY GATE (2026-10-16). Two corpora can have the SAME 12,000-word vocabulary SIZE and
completely different word lists. Comparing numbers across families is meaningless: token id 4 means a
different word in each. Measured: prepared_ext md5 01b1eba6f8 vs prepared_splitfix/prepared_clean
md5 82e0edd0. This script therefore derives each run's vocabulary family from the corpus its run-log
records, ranks ONLY within one family, and REFUSES any cross-family comparison (printing the md5s as
evidence) instead of silently printing an invalid ranking.

VERDICTS ARE COMPUTED, NEVER TYPED. The winner line and its caveats are derived from the parsed
numbers; no literal conclusion is printed that the arithmetic does not support.
"""
import hashlib
import json
import os
import re

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
SRC = os.path.join(PROJ, "lens/runs/MATRIX.txt")
OUT = os.path.join(PROJ, "lens/runs/MATRIX_SUMMARY.md")
RUNS = os.path.join(PROJ, "lens/runs/chat")


def vocab_md5(data_dir):
    """The vocabulary fingerprint of a corpus: same size is NOT the same word list."""
    if not data_dir or not os.path.isdir(data_dir):
        return None
    for name in sorted(os.listdir(data_dir)):
        if name.startswith("vocab") and name.endswith(".json"):
            with open(os.path.join(data_dir, name), "rb") as f:
                return hashlib.md5(f.read()).hexdigest(), name
    return None


def family_of_ship(ship):
    """Each shipped checkpoint's run-log records the corpus it trained on. Derive the family there."""
    if not ship or ship == "?":
        return {"family": None, "data_dir": None, "why": "no ship file named in MATRIX.txt"}
    out = re.sub(r"_ep\d+\.npz$", "", ship)
    log = os.path.join(RUNS, f"{out}_log.json")
    if not os.path.exists(log):
        return {"family": None, "data_dir": None, "why": f"no run log at {os.path.basename(log)}"}
    try:
        args = json.load(open(log)).get("args", {})
    except Exception as e:
        return {"family": None, "data_dir": None, "why": f"unreadable run log ({type(e).__name__})"}
    dd = args.get("data_dir")
    if not dd or dd == "(default)":
        return {"family": None, "data_dir": dd,
                "why": "run log records no corpus (run predates data_dir recording)"}
    fam = vocab_md5(dd)
    if not fam:
        return {"family": None, "data_dir": dd, "why": f"no vocabulary file under {dd}"}
    return {"family": fam[0], "data_dir": dd, "why": "ok", "vocab_file": fam[1]}


def parse(path):
    if not os.path.exists(path):
        return []
    blocks, cur = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("##########"):
            if cur:
                blocks.append(cur)
            cur = {"name": line.replace("##########", "").strip(), "lines": []}
        elif cur is not None:
            cur["lines"].append(line)
    if cur:
        blocks.append(cur)
    out = []
    for b in blocks:
        if "DONE" in b["name"]:
            continue
        txt = "\n".join(b["lines"])
        masked = re.findall(r"assistant-masked: acc ([0-9.]+)%\s+ppl ([0-9.]+)\s+\(n=(\d+)\)", txt)
        subj = re.search(r"SHIP (\S+)", txt)
        out.append({
            "name": b["name"],
            "ship": (subj.group(1) if subj else "?"),
            "merged_acc": float(masked[0][0]) if len(masked) > 0 else None,
            "merged_ppl": float(masked[0][1]) if len(masked) > 0 else None,
            "merged_n": int(masked[0][2]) if len(masked) > 0 else None,
            "own_acc": float(masked[1][0]) if len(masked) > 1 else None,
            "own_ppl": float(masked[1][1]) if len(masked) > 1 else None,
        })
    return out


def main():
    rows = parse(SRC)
    if not rows:
        print(f"  no matrix results yet in {SRC} (the matrix has not run or is still running)")
        return
    for r in rows:
        fam = family_of_ship(r["ship"])
        r.update({"family": fam["family"], "data_dir": fam["data_dir"], "family_why": fam["why"]})

    print("=" * 104)
    print("  COMPARISON MATRIX — ranked WITHIN one vocabulary family")
    print("=" * 104)
    families = {}
    for r in rows:
        families.setdefault(r["family"], []).append(r)
    for fam, members in families.items():
        label = "UNVERIFIED (no corpus recorded in the run log)" if fam is None else f"vocab md5 {fam}"
        dirs = sorted({m["data_dir"] for m in members if m["data_dir"]})
        print(f"  family {label}: {len(members)} variant(s)  corpora: {', '.join(dirs) or '?'}")
    print()

    comparable = [f for f in families if f is not None]
    if len(comparable) > 1:
        print("  REFUSED: the matrix spans MORE THAN ONE vocabulary family. Numbers across families")
        print("  are not comparable (the same token id is a different word), so no winner is named.")
        for fam, members in families.items():
            print(f"    - {fam}: {[m['name'] for m in members]}")
        return
    if len(comparable) == 0:
        print("  REFUSED: no variant's run log records the corpus it trained on, so no two numbers")
        print("  can be shown to share a vocabulary. Re-run the trainer (it now records data_dir).")
        for r in rows:
            print(f"    - {r['name']}: {r['family_why']}")
        return

    fam = comparable[0]
    ok = [r for r in families[fam] if r["merged_acc"] is not None]
    ok.sort(key=lambda r: -r["merged_acc"])
    print(f"  {'variant':<30} {'merged acc':>11} {'ppl':>9} {'own acc':>9} {'n':>7}  ship")
    for r in ok:
        own = f"{r['own_acc']:.2f}%" if r["own_acc"] is not None else "n/a"
        print(f"  {r['name']:<30} {r['merged_acc']:>10.2f}% {r['merged_ppl']:>9.2f} {own:>9} "
              f"{r['merged_n']:>7}  {r['ship']}")
    for r in [x for x in families[fam] if x["merged_acc"] is None]:
        print(f"  {r['name']:<30} {'INCOMPLETE':>11}  (no merged-val line found)")

    if len(ok) >= 2:
        win, second = ok[0], ok[1]
        margin = win["merged_acc"] - second["merged_acc"]
        print()
        print(f"  WINNER: {win['name']}  {win['merged_acc']:.2f}%  ppl {win['merged_ppl']:.2f}  "
              f"margin {margin:+.2f} pts over {second['name']}")
        print(f"  CAVEAT (computed, not typed): one seed per arm, no significance test -- a margin of "
              f"{margin:.2f} pts at n={win['merged_n']} is not shown to be real.")
    elif len(ok) == 1:
        print()
        print(f"  ONE COMPARABLE RUN: {ok[0]['name']} {ok[0]['merged_acc']:.2f}% -- nothing to rank it "
              f"against inside its vocabulary family.")

    ref = [r for r in rows if r["family"] != fam]
    if ref:
        print()
        print("  NOTE: variants outside this family are NOT compared (see the refusal above):")
        for r in ref:
            print(f"    - {r['name']} ({r['data_dir']}): {r['family_why']}")

    with open(OUT, "w") as f:
        f.write("# COMPARISON MATRIX — ranked WITHIN one vocabulary family\n\n")
        f.write(f"Vocabulary family: {fam} (md5 of the corpus's vocab file)\n\n")
        f.write("| variant | merged clean acc | ppl | own-corpus acc | n | ship |\n|---|---|---|---|---|---|\n")
        for r in ok:
            own = f"{r['own_acc']:.2f}%" if r["own_acc"] is not None else "n/a"
            f.write(f"| {r['name']} | {r['merged_acc']:.2f}% | {r['merged_ppl']:.2f} | {own} | "
                    f"{r['merged_n']} | {r['ship']} |\n")
        if len(ok) >= 2:
            f.write(f"\n**Winner (within family):** {ok[0]['name']} — {ok[0]['merged_acc']:.2f}% "
                    f"(margin {ok[0]['merged_acc'] - ok[1]['merged_acc']:+.2f} pts, single seed, "
                    f"no significance test)\n")
        f.write("\nCross-family comparisons are refused by this script: two corpora can share the "
                "12,000-word size and have entirely different word lists.\n")
    print(f"  written: {OUT}")


if __name__ == "__main__":
    main()