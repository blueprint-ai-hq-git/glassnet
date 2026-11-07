#!/usr/bin/env python3
"""Check every "value + file:line" citation on the public pages against the file it points at.

This project has twice published a correct number attached to the wrong line - once citing seed 1's line
numbering for a seed 2 result, once quoting an epoch path that did not exist in the log. Both were caught by
hand. This tool makes the check mechanical: for each table row or sentence on STATUS.md / README.md that carries
a percentage AND a `path:line` citation, the cited line must actually contain a number matching the quote.

What it does NOT check: whether the metric is the right one (VAL vs ASST - the P1 defect), or whether the run was
comparable at all. It only proves the line exists and contains the digits, which is the mechanical half.

Usage: check_public_cites.py [file ...]      default: STATUS.md README.md
Exit: 0 all cites resolve, 1 something does not, 2 a cited file is missing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/run/media/yahya/HARD DISK 4/GlassNet")
CITE = re.compile(r"`?((?:lens/runs|docs/history|data)[^`\s:,]*?\.(?:txt|md|tsv)):(\d+)`?")
PCT = re.compile(r"(\d+\.\d+)%")


def main() -> int:
    files = sys.argv[1:] or ["STATUS.md", "README.md"]
    total = ok = 0
    bad = []
    missing_files = set()
    for name in files:
        path = ROOT / name
        if not path.exists():
            print("missing page:", name)
            return 2
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            cites = CITE.findall(line)
            if not cites:
                continue
            pcts = PCT.findall(line)
            texts = []
            total += 1                     # one cited ROW (a row may carry several file:line cites)
            for rel, lineno in cites:
                target = ROOT / rel
                if not target.exists():
                    missing_files.add(rel)
                    bad.append((name, n, rel, lineno, "FILE MISSING", pcts))
                    continue
                try:
                    tline = target.read_text(errors="replace").splitlines()[int(lineno) - 1]
                except IndexError:
                    bad.append((name, n, rel, lineno, "LINE BEYOND END", pcts))
                    continue
                if not tline.strip():
                    bad.append((name, n, rel, lineno, "CITED LINE IS BLANK", pcts))
                    continue
                texts.append(tline)
            # a quoted number is supported if it appears on ANY cited line of that row
            union = "\n".join(texts)
            unsupported = [q for q in pcts if q not in union]
            if unsupported:
                if texts:
                    bad.append((name, n, cites[0][0], cites[0][1], "NO QUOTED VALUE ON ANY CITED LINE",
                                unsupported))
            elif texts:
                ok += 1

    print("cited rows checked: %d   fully resolved: %d   problems: %d" % (total, ok, len(bad)))
    for page, n, rel, lineno, why, pcts in bad:
        print("  %s:%d -> %s:%s  %s  quoted %s" % (page, n, rel, lineno, why,
                                                    ", ".join(p + "%" for p in pcts) or "(no number)"))
    if missing_files:
        print("missing cited files:", ", ".join(sorted(missing_files)))
    if bad:
        print("\nA row can legitimately quote a value that lives on a DIFFERENT line of the same file (e.g. a")
        print("final VAL and a best-checkpoint ASST). Confirm each problem by hand before editing anything.")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
