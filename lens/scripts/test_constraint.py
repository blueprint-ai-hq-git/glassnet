"""test_constraint.py — prove the SyntaxConstraint works BEFORE wiring it to a model.

Checks, with CPython's own ast.parse as ground truth:
  * 3 REAL python programs must be ACCEPTED
  * 4 BROKEN variants (missing paren, dangling operator, bad indent, unclosed bracket) must be
    REFUSED
  * the token-level `allows()` query must agree with the whole-program verdict on the same input
If this fails, the constraint is wrong and must not be wired to the model.
"""
import ast
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
from constrain_code import SyntaxConstraint, mask_invalid   # noqa: E402

OK_SRC = [
    "def add ( a , b ) : return a + b\n",
    "class BankAccount : \n def __init__ ( self ) : self . balance = 0\n",
    "for i in range ( 10 ) : print ( i )\n",
]
BAD_SRC = [
    "def add ( a , b : return a + b\n",          # ( never closed
    "return a + \n",                              # dangling operator then newline
    "def f ( ) : \n return 1\n  x = 2\n",          # dedent to a level never opened
    "x = [ 1 , 2\n",                              # [ never closed
]

def _py_ok(s):
    try:
        ast.parse(s)
        return True
    except SyntaxError:
        return False


print("=" * 90)
print("SYNTAX CONSTRAINT TEST — accepted/refused counts, with CPython as ground truth")
print("=" * 90)


def _tokens(src):
    """Keep whitespace as tokens. `.split()` would drop every space, turning 'def add' into
    ['def','add'] which joins to 'defadd' -- a mangled program no checker could accept."""
    import re as _re
    return _re.findall(r"\s+|\S+", src)


def verdict(src):
    c = SyntaxConstraint()
    for t in _tokens(src):
        if not c.feed(t):
            return "REFUSED"
    return "ACCEPT" if c.valid_so_far() else "REFUSED"


print("\n  REAL programs (want ACCEPT):")
ok = 0
for s in OK_SRC:
    v = verdict(s)
    truth = "ACCEPT" if _py_ok(s) else "REFUSED"
    mark = "OK" if v == truth == "ACCEPT" else "WRONG"
    ok += 1 if mark == "OK" else 0
    print(f"    {v:<9} (python: {truth})  {mark}  {s[:52]!r}")

print("\n  BROKEN programs (want REFUSED):")
bad = 0
for s in BAD_SRC:
    v = verdict(s)
    truth = "ACCEPT" if _py_ok(s) else "REFUSED"
    mark = "OK" if v == truth == "REFUSED" else "WRONG"
    bad += 1 if mark == "OK" else 0
    print(f"    {v:<9} (python: {truth})  {mark}  {s[:52]!r}")

print("\n" + "-" * 90)
print(f"  real accepted correctly: {ok}/{len(OK_SRC)}   broken refused correctly: "
      f"{bad}/{len(BAD_SRC)}")


def _py_ok(s):
    try:
        ast.parse(s)
        return True
    except SyntaxError:
        return False


# token-level query must agree with the whole-program verdict
print("\n  token-level agrees() check (must not disagree with CPython on these cases):")
agree = total = 0
for s in OK_SRC + BAD_SRC:
    c = SyntaxConstraint()
    toks = s.replace("\n", " \n ").split()
    dis = 0
    for t in toks:
        if not c.allows(t) and _py_ok(s):
            dis += 1
        c.feed(t)
    total += 1
    agree += 1 if dis == 0 else 0
    print(f"    {s[:50]!r:<54} disagreements {dis}")
print(f"\n  {agree}/{total} cases with zero disagreements")

passed = ok == len(OK_SRC) and bad == len(BAD_SRC) and agree == total
print("\n  " + ("PASS — the constraint is sound; safe to wire to the model." if passed
               else "FAIL — fix the constraint BEFORE using it on the model."))
sys.exit(0 if passed else 1)
