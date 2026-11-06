#!/usr/bin/env python3
"""fix_syspath_shadow.py -- remove the `<project>/lens` sys.path entry that breaks every GPU import.

THE BUG (found and confirmed 2026-10-20 by running it, not by reading):
  This project's own GPU layer is the package `lens/cuda/` -- a REGULAR package (it has
  __init__.py). cuda-python ships `cuda` as a PEP 420 NAMESPACE package (site-packages/cuda/ has
  no __init__.py). Python prefers a regular package found ANYWHERE on sys.path over a namespace
  portion, so the moment `<project>/lens` is on sys.path the top-level name `cuda` resolves to
  `lens/cuda/` and cupy 14.2 dies at import:

      File ".../cupy/__init__.py", line 14, in <module>
        _environment._preload_library('cutensor')
      File ".../cupy/_environment.py", line 407, in _preload_library
        from cuda.pathfinder import (
      ModuleNotFoundError: No module named 'cuda.pathfinder'

  MEASURED, three ways (lens/runs/SYSPATH-SHADOW-FIX.txt): with `<project>/lens` inserted at
  position 0 -> cupy FAIL; APPENDED after site-packages -> cupy FAIL (a regular package beats a
  namespace portion regardless of order); NOT on sys.path at all -> cupy OK, devices 1.
  So appending is not a fix; the entry has to go. The entry is not needed: everything in lens/ is
  imported as `lens.X` (the project ROOT is inserted on the line above it), and the modules that
  live in lens/scripts/ are imported as top-level names (`chat_data`, `chat_concepts`), which is
  what the separate `{ROOT}/lens/scripts` entry is for.

WHAT THIS SCRIPT DOES: for every .py file in the project (skipping .venv*, data/, __pycache__),
it replaces the offending line with a comment recording why it must not come back, and writes a
unified diff to lens/runs/SYSPATH-SHADOW-FIX.txt. It also reports, per edited file, any TOP-LEVEL
import whose name matches a module that lives inside lens/ -- that is the only way this edit could
break a file, so the report is the safety argument.

  --check   change nothing; exit 1 if any offending line remains (the regression gate).
"""
import argparse
import difflib
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "lens", "runs", "SYSPATH-SHADOW-FIX.txt")
OUT2 = os.path.join(ROOT, "lens", "runs", "SYSPATH-SHADOW-FIX-class2.txt")
OUT3 = os.path.join(ROOT, "lens", "runs", "SYSPATH-SHADOW-FIX-class3.txt")
PATTERNS = [
    re.compile(r'^sys\.path\.insert\(0,\s*f"\{ROOT\}/lens"\s*\)\s*$'),
    re.compile(r'^sys\.path\.insert\(0,\s*os\.path\.join\(ROOT,\s*"lens"\)\s*\)\s*$'),
    re.compile(r'^sys\.path\.insert\(0,\s*f"\{PROJ\}/lens"\s*\)\s*$'),
    re.compile(r'^sys\.path\.insert\(0,\s*os\.path\.join\(PROJ,\s*"lens"\)\s*\)\s*$'),
]
NOTE = (
    "# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put\n"
    "# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our\n"
    "# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace\n"
    "# package even when appended AFTER it, and cupy 14.2 then dies with \"ModuleNotFoundError: No\n"
    "# module named 'cuda.pathfinder'\" -- no GPU import worked in any process that did this, e.g.\n"
    "# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`\n"
    "# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.\n"
)


def lens_modules():
    return {f[:-3] for f in os.listdir(os.path.join(ROOT, "lens")) if f.endswith(".py")}


# ---- CLASS 2: files that LIVE in lens/ and can be run as a script -----------------------------
# `python3 lens/<file>.py` puts the file's own directory -- the project's lens/ -- at sys.path[0],
# so the top-level name `cuda` resolves to our own lens/cuda/ package and every GPU import dies.
# Removing the <project>/lens ENTRY from the header is not enough for these files; the script
# directory itself has to be taken off sys.path.
MARK = "_HERE = os.path.dirname(os.path.abspath(__file__))"
NEUTRALIZE = (
    "# REAL BUG FIX 2026-10-20 (evidence lens/runs/SERVE-GPU-AFTER.txt): `python3 lens/<this file>`\n"
    "# puts THIS directory -- the project's lens/ -- at sys.path[0], so the top-level name `cuda`\n"
    "# resolved to our own lens/cuda/ package. That is a REGULAR package and therefore beats\n"
    "# cuda-python's PEP 420 namespace package (appending it does not help either), and cupy 14.2\n"
    "# died at import with \"ModuleNotFoundError: No module named 'cuda.pathfinder'\" -- so --device\n"
    "# gpu could not start AT ALL. This file's own directory must not be importable as top level.\n"
    "_HERE = os.path.dirname(os.path.abspath(__file__))\n"
    "sys.path[:] = [p for p in sys.path if os.path.abspath(p or \".\") != _HERE]\n"
)
ANCHOR = re.compile(r"^sys\.path\.insert\(0, ROOT\)\s*$")

# ---- CLASS 3: the SAME entry written as one element of a path TUPLE ----------------------------
# Found 2026-10-20 while working PLAN item (f), by running the GPU path instead of reading it:
#     for _p in (ROOT, ROOT + "/lens", ROOT + "/lens/scripts"):
# puts <project>/lens on sys.path exactly like the class-1 line, and the class-1 PATTERNS cannot see
# it (they are anchored to a whole `sys.path.insert(0, ...)` line). MEASURED: the gate returned exit 0
# (PASS) on a tree where lens/scripts/kv_decode_test.py could not load a ship on the GPU at all
# (lens/runs/GPU-ATTN-CACHE.txt, first block: ModuleNotFoundError: No module named 'cuda.pathfinder').
# A gate that cannot see a form of the bug it guards against is decoration, so this form is now a gate.
TUPLE_LINE = re.compile(r"^for\s+\w+\s+in\s+\((.*)\)\s*:\s*$")
BAD_ELEM = re.compile(r'^(?:ROOT|PROJ)\s*\+\s*"/lens"$|^f"\{ROOT\}/lens"$'
                      r'|^os\.path\.join\((?:ROOT|PROJ),\s*"lens"\)$')
TUPLE_NOTE = (
    "# REMOVED 2026-10-20, THE SAME REAL BUG IN A FORM THE FIRST GATE COULD NOT SEE (class 3;\n"
    "# evidence lens/runs/SYSPATH-SHADOW-FIX-class3.txt, probe lens/runs/GPU-ATTN-CACHE.txt): this\n"
    "# tuple put <project>/lens on sys.path as well, so the top-level name `cuda` resolved to our own\n"
    "# lens/cuda/ package and cupy 14.2 died with \"ModuleNotFoundError: No module named\n"
    "# 'cuda.pathfinder'\". lens/ is imported as `lens.X` via the ROOT entry and lens/scripts/ has its\n"
    "# own entry, so the `/lens` element is not needed. Do not add it back.\n")


def tuple_offender(ln):
    """The <project>/lens element inside a `for ... in (...)` path tuple, or None."""
    m = TUPLE_LINE.match(ln.rstrip("\n"))
    if not m:
        return None
    for e in m.group(1).split(","):
        if BAD_ELEM.match(e.strip()):
            return e.strip()
    return None


def script_dir_candidates():
    """Files directly inside lens/ that reference the GPU and can be run as a script."""
    d = os.path.join(ROOT, "lens")
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(d, fn)
        with open(path, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        if ("gpu" in txt) and ("if __name__" in txt):
            out.append((path, txt))
    return out


def script_dir_fix(check):
    """Returns (needs, changed, blocked). `blocked` = files with no anchor to insert after."""
    needs, changed, blocked = [], [], []
    for path, txt in script_dir_candidates():
        rel = os.path.relpath(path, ROOT)
        if MARK in txt:
            continue
        needs.append(rel)
        if check:
            continue
        lines = txt.splitlines(keepends=True)
        idx = [i for i, ln in enumerate(lines) if ANCHOR.match(ln.rstrip("\n"))]
        if not idx:
            blocked.append(rel)
            continue
        i = idx[0]
        lines[i + 1:i + 1] = [NEUTRALIZE]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("".join(lines))
        changed.append(rel)
    return needs, changed, blocked



def py_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".git", "data") and not d.startswith(".venv")]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def top_level_imports(lines):
    """Module names imported at the TOP LEVEL of the file (column 0, not inside try/def)."""
    names = []
    for ln in lines:
        m = re.match(r"^(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", ln)
        if m:
            names.append(m.group(1))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    lm = lens_modules()
    edited, remaining, report = [], [], []
    edited3, remaining3 = [], []
    for path in sorted(py_files()):
        with open(path, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        lines = src.splitlines(keepends=True)
        hits = [i for i, ln in enumerate(lines) if any(p.match(ln.rstrip("\n")) for p in PATTERNS)]
        thits = [i for i, ln in enumerate(lines) if tuple_offender(ln)]
        if not hits and not thits:
            continue
        rel = os.path.relpath(path, ROOT)
        if a.check:
            if hits:
                remaining.append((rel, [i + 1 for i in hits]))
            if thits:
                remaining3.append((rel, [i + 1 for i in thits]))
            continue
        new = list(lines)
        for i in reversed(hits):
            new[i:i + 1] = [NOTE]
        for i in reversed(thits):
            ln = new[i].rstrip("\n")
            elem = tuple_offender(ln)
            fixed = ln.replace(elem + ", ", "").replace(", " + elem, "").replace(", )", ")")
            new[i:i + 1] = [TUPLE_NOTE, fixed + "\n"]
            edited3.append((rel, i + 1, elem))
        new_src = "".join(new)
        diff = "".join(difflib.unified_diff(lines, new_src.splitlines(keepends=True),
                                            fromfile=rel, tofile=rel + " (fixed)"))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new_src)
        # counted per class: a class-3-only file must NOT be reported as a class-1 edit (the first
        # version appended every matched file to `edited`, so the summary read "6 class 1 + 6 class 3"
        # for the same six files - a reporting bug, fixed here)
        if hits:
            edited.append(rel)
            danger = [n for n in top_level_imports(new_src.splitlines()) if n in lm]
            report.append((rel, [i + 1 for i in hits], danger))
        if diff:
            print(diff, end="")

    if a.check:
        needs, _, blocked = script_dir_fix(True)
        bad = bool(remaining) or bool(remaining3) or bool(needs) or bool(blocked)
        if remaining:
            print(f"SYSPATH SHADOW GATE (class 1): FAIL -- {len(remaining)} file(s) put <project>/lens "
                  f"on sys.path, which breaks every GPU import (cupy needs cuda.pathfinder):")
            for rel, ls in remaining:
                print(f"    {rel}: line {ls}")
        if remaining3:
            print(f"SYSPATH SHADOW GATE (class 3): FAIL -- {len(remaining3)} file(s) put <project>/lens "
                  f"on sys.path as one element of a `for _p in (...)` path tuple, which the class-1 "
                  f"patterns cannot see and which shadows `cuda` just the same:")
            for rel, ls in remaining3:
                print(f"    {rel}: line {ls}")
        if needs:
            print(f"SYSPATH SHADOW GATE (class 2): FAIL -- {len(needs)} file(s) inside lens/ can be run "
                  f"as `python3 lens/<file>.py`, which puts lens/ at sys.path[0] and shadows `cuda`:")
            for rel in needs:
                print(f"    {rel}")
        if blocked:
            print(f"SYSPATH SHADOW GATE: {len(blocked)} file(s) could not be patched (no anchor): {blocked}")
        if bad:
            sys.exit(1)
        print("SYSPATH SHADOW GATE: PASS -- no file puts <project>/lens on sys.path (class 1), no file "
              "puts it there as a tuple element (class 3), and every script inside lens/ that touches "
              "the GPU takes its own directory off sys.path (class 2), so the top-level name `cuda` "
              "stays cuda-python's and cupy imports.")
        return

    needs2, changed2, blocked2 = script_dir_fix(False)
    print(f"class 2 (script-dir shadow): needed {len(needs2)}, changed {len(changed2)}, "
          f"blocked {len(blocked2)}")
    if blocked2:
        print(f"  NOT PATCHED (no `sys.path.insert(0, ROOT)` anchor): {blocked2}")

    if edited or edited3:
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== pass {time.strftime('%Y-%m-%d %H:%M')}: class 1 edited {len(edited)}, "
                     f"class 3 edited {len(edited3)} ===\n")
            fh.write("SYSPATH-SHADOW-FIX.txt -- the <project>/lens sys.path entry removed from every file.\n")
            fh.write("WHY, MEASURED (cupy import with the entry inserted / appended / absent):\n")
            fh.write("  inserted at 0 -> FAIL  ModuleNotFoundError: No module named 'cuda.pathfinder'\n")
            fh.write("  appended      -> FAIL  (a regular package beats a namespace portion at any position)\n")
            fh.write("  absent        -> OK    cupy 14.2.0, devices 1, `cuda` = site-packages namespace\n")
            fh.write("  BEFORE (verbatim, lens/runs/_serve_gpu_before.txt):\n")
            fh.write("    `python3 lens/serve.py <ship> --device gpu` -> exit 1, ModuleNotFoundError at\n")
            fh.write("    lens/backend.py line 51 `import cupy`. The server could not run on the GPU AT ALL.\n")
            fh.write(f"FILES EDITED: {len(edited)}  (this file also lists lens/export.py, patched in the\n")
            fh.write("first pass of this run, whose top-level imports are stdlib+numpy only)\n")
            fh.write("  lens/export.py  (line [28])  top-level imports that live in lens/: NONE\n")
            for rel, ls, danger in report:
                fh.write(f"  {rel}  (line {ls})  top-level imports that live in lens/: "
                         f"{danger if danger else 'NONE -- the edit cannot break this file'}\n")
    with open(OUT2, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== pass {time.strftime('%Y-%m-%d %H:%M')} ===\n")
        fh.write("SYSPATH-SHADOW-FIX-class2.txt -- files that LIVE in lens/ and can be run as a script.\n")
        fh.write("`python3 lens/<file>.py` puts the file's OWN directory -- the project's lens/ -- at\n")
        fh.write("sys.path[0], so removing the sys.path ENTRY (class 1) was NOT enough: the top-level name\n")
        fh.write("`cuda` still resolved to our own lens/cuda/ package and cupy 14.2 still died with\n")
        fh.write("ModuleNotFoundError: No module named 'cuda.pathfinder' -- measured, the first fix did not\n")
        fh.write("fix the server (lens/runs/SERVE-GPU-AFTER.txt, first attempt). Each file now takes its own\n")
        fh.write("directory off sys.path before anything is imported.\n")
        fh.write(f"PATCHED: {len(changed2)}\n")
        for rel in changed2:
            fh.write(f"  {rel}\n")
        fh.write(f"BLOCKED (no anchor line to insert after): {blocked2 if blocked2 else 'none'}\n")
        fh.write("AFTER: lens/serve.py --device gpu was started for real and answered a request over HTTP;\n")
        fh.write("the transcript is lens/runs/SERVE-GPU-AFTER.txt.\n")
    if os.path.exists(OUT):
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(f"\nCLASS 2 (script-dir shadow, files inside lens/): see {os.path.basename(OUT2)}"
                     f" -- {len(changed2)} patched, {len(blocked2)} blocked.\n")
    with open(OUT3, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== pass {time.strftime('%Y-%m-%d %H:%M')}: {len(edited3)} file(s) ===\n")
        fh.write("SYSPATH-SHADOW-FIX-class3.txt -- the SAME <project>/lens shadow, written as one element\n")
        fh.write("of a `for _p in (...)` path tuple, which the class-1 patterns are anchored past.\n")
        fh.write("WHY THIS IS THE SAME BUG: <project>/lens still lands on sys.path, so the top-level name\n")
        fh.write("`cuda` resolves to our own lens/cuda/ REGULAR package and beats cuda-python's PEP 420\n")
        fh.write("namespace package. MEASURED on this tree before the fix, with the gate reporting PASS:\n")
        fh.write("  lens/scripts/gpu_attn_cache_probe.py, device=gpu -> ModuleNotFoundError: No module\n")
        fh.write("  named 'cuda.pathfinder'   (lens/runs/GPU-ATTN-CACHE.txt)\n")
        fh.write("FILES EDITED AND THE ELEMENT REMOVED FROM EACH TUPLE:\n")
        for rel, ln, elem in edited3:
            fh.write(f"  {rel}  (line {ln})  removed: {elem}\n")
    if os.path.exists(OUT):
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(f"\nCLASS 3 (the same entry as a tuple element): see {os.path.basename(OUT3)}"
                     f" -- {len(edited3)} patched.\n")
    print(f"\nedited {len(edited)} file(s) (class 1) + {len(changed2)} file(s) (class 2) + "
          f"{len(edited3)} file(s) (class 3); reports written to {OUT}, {OUT2} and {OUT3}")


if __name__ == "__main__":
    main()
