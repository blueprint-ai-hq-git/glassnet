"""selfctx_preflight_grade.py — grades the self-context wiring pre-flight from the stored text of the five tiny
runs (predictions P98a-c, lens/runs/SELF-CONTEXT-PREDICTIONS-P98.txt). Text only: no checkpoint, no model, no GPU.

Prints the VALUES IT COMPARED beside every verdict (this project has three recorded verdict-check bugs where a
right number was given a wrong conclusion).
"""
import os
import re
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"
TMP = os.path.join(P, "lens/runs/_tmp")


def read(tag):
    with open(os.path.join(TMP, f"sctxpf_{tag}.out")) as f:
        return f.read()


def epoch_line(txt):
    for ln in txt.splitlines():
        if re.match(r"\s*ep\s+\d+/\d+", ln) and "loss" in ln:
            return ln.strip()
    return None


def epoch_core(txt):
    """The epoch line WITHOUT the trailing wall-time field.

    'digit for digit' has to mean every field EXCEPT '(NNs)': the second field is wall-clock time, and the two
    control runs of the 2026-10-25 block printed 49s and 51s on otherwise identical lines. Comparing the whole
    line made the P98a control read FAIL for a timing difference - a verdict about the model drawn from the
    machine's speed. The fields compared are printed beside the verdict either way.
    """
    ln = epoch_line(txt)
    if ln is None:
        return None
    return re.sub(r"\s*\(\d+s\)\s*$", "", ln)


def scoped(txt, tag):
    """Only the text of the LAST run of this tag (the per-run files are fresh per launch since the same fix)."""
    i = txt.rfind(f"ARM {tag}")
    return txt[i:] if i >= 0 else txt


def exit_of(txt):
    m = re.findall(r"^exit=(\d+)", txt, re.M)
    return int(m[-1]) if m else None


def sctx_lines(txt):
    return [l.strip() for l in txt.splitlines() if "SELF-CONTEXT" in l]


pre = read("prechange")
new = read("newdefault")
on = read("selfctx_on")
g_mtp = read("guard_mtp")
g_rnd = read("guard_rounds")

ep_pre, ep_new, ep_on = epoch_line(pre), epoch_line(new), epoch_line(on)
ex_pre, ex_new, ex_on = exit_of(pre), exit_of(new), exit_of(on)
ex_mtp, ex_rnd = exit_of(g_mtp), exit_of(g_rnd)

print()
print("P98a CONTROL - THE FLAG-ABSENT PATH IS UNCHANGED (new trainer vs the pre-change copy)")
print("  comparing every printed field EXCEPT the trailing wall-time '(NNs)'; both exits and both lines shown:")
print(f"  pre-change epoch line: {ep_pre}")
print(f"  new file   epoch line: {ep_new}")
print(f"  exits: pre-change={ex_pre}  new={ex_new}")
print(f"  SELF-CONTEXT lines in the flag-absent new run: {len(sctx_lines(scoped(new, 'newdefault')))} (predicted 0)")
core_pre, core_new = epoch_core(pre), epoch_core(new)
same = core_pre is not None and core_pre == core_new
actually_crashed = (ex_new != 0) or re.search(r"Traceback", scoped(new, "newdefault")) is not None
print(f"  field-wise identical (wall time excluded): {same}")
print(f"  the flag-absent run ended cleanly: {not actually_crashed} (exit={ex_new})")
if actually_crashed:
    # 2026-10-25, FIRST PRE-FLIGHT BLOCK: the new file died with
    # "TypeError: ChatNet.loss() got an unexpected keyword argument 'concept_ids'" because the wrapper class
    # inside train_chat.py forwards to the library's loss with a fixed signature. The first version of THIS
    # grader then printed "the default path changed - revert it", which is a WRONG reading of a CRASH: nothing
    # about the flag-absent path was shown to change, the run simply did not finish. Verdicts must name what
    # happened (BUGS.md, verdict-check class).
    print("  -> P98a INVALID CONTROL RUN - the flag-absent arm DID NOT COMPLETE")
    print("     NOT a 'default path changed' reading: fix the crash and re-run the pre-flight, then read the LAST")
    print("     block of this file.")
elif same:
    bad = sctx_lines(scoped(new, "newdefault"))
    if bad:
        print("  -> P98a FAIL: the flag-absent run printed SELF-CONTEXT lines, which it must not")
    else:
        print("  -> P98a PASS")
else:
    print("  -> P98a FAIL: the flag-absent epoch line DIFFERS in a real field:")
    for a, b in zip((core_pre or "").split(), (core_new or "").split()):
        if a != b:
            print(f"       pre-change {a!r}  vs  new {b!r}")
    print("     The edit changed the default path. It is to be REVERTED or repaired, not explained (P98a).")

print()
print("P98b THE LEVER IS REAL AND MEASURED (--self-context-frac 1.0 --self-context-rounds 2)")
print(f"  self-context run epoch line: {ep_on}")
print(f"  differs from the flag-absent line: {ep_on is not None and ep_on != ep_new} (must be True: a change that")
print("    changes nothing is a silent no-op)")
for l in sctx_lines(on)[:4]:
    print("  trainer: " + l)
m = re.search(r"round-1 own-token agreement with the TRUE next token\s+([\d,]+)/([\d,]+)\s+=\s+([\d.]+)%", on)
if m:
    hit, n, pct = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")), float(m.group(3))
    # the header's fraction belongs to the first (only) epoch here; read the numbers as printed
    strict = 0.0 < pct < 100.0
    print(f"  round-1 agreement, as printed: {hit:,}/{n:,} = {pct:.2f}%  (strictly between 0 and 100: {strict})")
    print(f"  -> P98b {'PASS' if (strict and ex_on == 0 and ep_on != ep_new) else 'FAIL'}")
else:
    print("  -> P98b CANNOT GRADE: no SELF-CONTEXT per-epoch line with an agreement figure was found.")

print()
print("P98c THE GUARDS FIRE BEFORE ANY WORK IS DONE")
for nm, txt, ex, want in (("--mtp 2", g_mtp, ex_mtp, "requires --mtp 1"),
                          ("--self-context-rounds 99", g_rnd, ex_rnd, "must be in 1..31")):
    msg = [l.strip() for l in txt.splitlines() if "refusing" in l.lower() or "must be in" in l or "requires" in l]
    m2 = re.search(r"SystemExit: (.*)", txt)
    print(f"  {nm}: exit={ex}  (predicted non-zero and no epoch line)")
    print(f"    message: {(m2.group(1).strip() if m2 else (msg[-1] if msg else 'NONE FOUND'))[:150]}")
    print(f"    epoch line written: {epoch_line(txt) is not None} (must be False)")
    ok = (ex is not None and ex != 0) and epoch_line(txt) is None and (want.split()[-1] in txt or want in txt)
    print(f"    -> {'PASS' if ok else 'FAIL'}")
print()
print("NOTE: these five runs are a WIRING test at d=64. No accuracy, perplexity or talk number from them may be")
print("quoted as a result about the model, and none is.")
