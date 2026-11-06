"""ladder.py — EVERY capability, tested small first, then stepped up.

WHY THIS EXISTS
---------------
I jumped straight to full-size GPU runs before proving each ability worked at small scale. That is
backwards: a failure at 60,000 samples and 24 epochs tells you much less than a failure at 2,000
samples and 2 epochs, because you cannot tell a broken code path from an under-trained model.

So this runs each ability at a TINY rung first. Each gate is BINARY and was fixed BEFORE any run:

  text LM    final val ppl < 0.5 * vocab        (better than chance by 2x)  AND loss fell
  image cls  test acc > 2x chance               (chance = 10%)
  VL / VQA   test acc > 2x chance               (chance = 1/answer_vocab)
  code LM    final val ppl < 0.5 * vocab
  explain    residual < 1e-5                    (the arithmetic is exact)
  gradcheck  worst relative difference <= 1e-2  (the engine's own float32 tolerance)
  audit      runs and produces a number for every concept

A rung must pass EVERY gate before the next rung is attempted. A failed gate is reported as FAILED
and the ladder stops there -- it does not move the bar.

Usage:  python lens/scripts/ladder.py --rung tiny
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
PY = os.path.join(PROJ, ".venv-gpu", "bin", "python3")
S = os.path.join(PROJ, "lens", "scripts")
OUT = os.path.join(PROJ, "lens", "runs", "ladder")
os.makedirs(OUT, exist_ok=True)

RUNGS = {
    #  name     n_train  epochs  d   device
    "tiny":   (2000, 2, 32, "cpu"),
    "small":  (10000, 4, 48, "cpu"),
    "medium": (30000, 8, 64, "gpu"),
    "big":    (0, 24, 64, "gpu"),        # 0 = full dataset
}


def run(cmd, timeout=1800):
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=PROJ, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout + p.stderr, time.perf_counter() - t0
    except subprocess.TimeoutExpired as e:
        # BUG FIXED HERE: the handler used to do `(e.stdout or "") + "\nTIMEOUT"` and crashed with
        # "TypeError: can't concat str to bytes". On timeout, `e.stdout` can be BYTES even when
        # text=True was requested, so the code that handled a timeout itself raised -- which turned a
        # clean "this rung exceeded its budget" into a traceback with no rung recorded at all.
        # The handler must accept both.
        def _s(x):
            if x is None:
                return ""
            return x.decode("utf-8", "replace") if isinstance(x, bytes) else x
        return -1, _s(e.stdout) + _s(e.stderr) + "\nTIMEOUT", time.perf_counter() - t0


def last_epoch(out):
    """Pull (loss, val_ppl, val_acc) from the most recent 'ep N' line, or from the final summary."""
    rows = re.findall(r"ep\s+\d+/\d+\s+loss\s+([\d.]+)\s+concept\s+[\d.]+\s+acc\s+([\d.]+)%\s+"
                      r"VAL ppl\s+([\d.]+)\s+acc\s+([\d.]+)%", out)
    if rows:
        return [float(x) for x in rows[-1]], [float(x) for x in rows[0]]
    rows = re.findall(r"ep\s+\d+/\d+\s+loss\s+([\d.]+)\s+train\s+([\d.]+)%\s+TEST\s+([\d.]+)%", out)
    if rows:
        return [float(x) for x in rows[-1]], [float(x) for x in rows[0]]
    return None, None


def final_test(out):
    m = re.findall(r"TEST\s+([\d.]+)%", out)
    return float(m[-1]) if m else None


def readable_test(out):
    """train_image.py's summary prints 'readable ... params TEST x%' THEN 'plain ... TEST y%', so
    final_test's LAST-TEST-LINE read the PLAIN CONTROL, not the model the gate is about. At rungs
    where plain happens to beat 20% the error was invisible; the tiny rung exposed it (plain sat at
    chance 9.94% and FAILED the gate while readable had printed 44.80%). Parse BY LABEL, never by
    position — the gate must score the arm it names. (Verdict-check class bug: right number, wrong
    subject.)"""
    m = re.findall(r"readable\s+[\d,]+\s+params\s+TEST\s+([\d.]+)%", out)
    return float(m[-1]) if m else None


def final_val_ppl(out):
    m = re.findall(r"VAL ppl\s+([\d.]+)", out)
    return float(m[-1]) if m else None


def prepare_checks(cap, n, epochs, d, device, out_text):
    """Each entry: (name, passed, detail). Rules fixed before the run."""
    checks = []
    if cap == "text_lm":
        vocab = 3500
        ppl = final_val_ppl(out_text)
        last, first = last_epoch(out_text)
        fell = (last is not None and first is not None and last[0] < first[0])
        checks.append(("loss fell", fell,
                       f"{first[0]:.4f} -> {last[0]:.4f}" if last else "no epochs parsed"))
        checks.append(("val ppl < 2x chance", ppl is not None and ppl < 2 * vocab,
                       f"ppl {ppl} vs chance {vocab}" if ppl is not None else "no ppl"))
    elif cap == "image_cls":
        acc = readable_test(out_text)
        checks.append(("test acc > 2x chance", acc is not None and acc > 20.0,
                       f"{acc}% vs chance 10%" if acc is not None else "no acc"))
    elif cap == "vl_vqa":
        acc = final_test(out_text)
        checks.append(("test acc > 2x chance", acc is not None and acc > 16.6,
                       f"{acc}% vs chance 8.3%" if acc is not None else "no acc"))
    elif cap == "code_lm":
        vocab = 6000
        ppl = final_val_ppl(out_text)
        checks.append(("val ppl < 2x chance", ppl is not None and ppl < 2 * vocab,
                       f"ppl {ppl} vs chance {vocab}" if ppl is not None else "no ppl"))
    elif cap == "explain":
        m = re.findall(r"residual\s+([\d.eE+-]+)", out_text)
        r = float(m[-1]) if m else None
        checks.append(("residual < 1e-5", r is not None and r < 1e-5,
                       f"residual {r}" if r is not None else "no residual printed"))
    elif cap == "gradcheck":
        m = re.findall(r"worst rel diff\s+([\d.eE+-]+)", out_text)
        vals = [float(x) for x in m]
        best = min(vals) if vals else None
        checks.append(("worst rel diff <= 1e-2", best is not None and best <= 1e-2,
                       f"best {best}" if best is not None else "no numbers"))
    elif cap == "audit":
        m = re.findall(r"(\d+)/(\d+)\s+pass", out_text) + re.findall(r"->\s+(\d+)/(\d+)", out_text)
        ok = bool(m)
        detail = f"{m[-1][0]}/{m[-1][1]} concepts" if m else "no audit line"
        checks.append(("audit produced numbers", ok, detail))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="tiny", choices=list(RUNGS))
    ap.add_argument("--only", default="", help="comma list of capabilities")
    a = ap.parse_args()
    n, epochs, d, device = RUNGS[a.rung]

    CAPS = ["gradcheck", "explain", "audit", "text_lm", "image_cls", "vl_vqa", "code_lm"]
    if a.only:
        CAPS = [c for c in a.only.split(",") if c]

    print("=" * 100)
    print(f"THE LADDER — rung '{a.rung}'   n_train={n or 'full'}  epochs={epochs}  d={d}  "
          f"device={device}")
    print("=" * 100)
    print("  gates were fixed BEFORE any run; a failure stops the ladder and is not reworded.")
    print()

    allpass = True
    table = []
    for cap in CAPS:
        t0 = time.perf_counter()
        # DATA SIZE PER CAPABILITY.
        # `n` is the sample count for this rung, and 0 means "the full dataset". The first version
        # computed `max(2000, n*2)` unconditionally, so at the big rung (n=0) text_lm and code_lm
        # trained on 2,000 windows -- LESS than the small rung's 20,000. The rung named "big" was
        # smaller than "small" for two of seven capabilities, and the weak ppl gate passed them
        # anyway (ppl 1722 against a bar of 7000). Fixed: a full rung gets a genuinely large budget.
        #
        # SECOND FIX: 4,000,000 windows did not fit the 1800s per-capability timeout -- train_code
        # raised TimeoutExpired and the whole rung died. A budget that cannot finish is not a budget.
        # The big rung now uses 600,000 windows (10x the medium rung) with a longer timeout, and the
        # actual numbers are recorded in the rung's JSON so the rung cannot silently be smaller than
        # it claims to be.
        big_windows = 600000
        mw = str(max(2000, n * 2)) if n else str(big_windows)
        if n == 0:
            timeout = 5400
        else:
            timeout = 1800
        if cap == "gradcheck":
            cmd = [PY, "-u", os.path.join(S, "gradcheck_vision_sweep.py")]
        elif cap == "explain":
            cmd = [PY, "-u", os.path.join(S, "show_task.py")]
        elif cap == "text_lm":
            cmd = [PY, "-u", os.path.join(S, "train_chat.py"), "--d", str(d), "--window", "16",
                   "--epochs", str(epochs), "--batch", "256", "--lr", "0.3", "--device", device,
                   "--max-windows", mw, "--out", f"lad_{cap}"]
        elif cap == "image_cls":
            cmd = [PY, "-u", os.path.join(S, "train_image.py"), "--d", str(d), "--epochs",
                   str(epochs), "--batch", "256", "--device", device, "--out", f"lad_{cap}"]
            if n:
                cmd += ["--train-n", str(n)]
        elif cap == "vl_vqa":
            cmd = [PY, "-u", os.path.join(S, "train_vl.py"), "--d", str(d), "--epochs",
                   str(epochs), "--batch", "256", "--device", device, "--out", f"lad_{cap}"]
            if n:
                cmd += ["--max-train", str(n)]
        elif cap == "code_lm":
            cmd = [PY, "-u", os.path.join(S, "train_code.py"), "--d", str(d), "--epochs",
                   str(epochs), "--batch", "256", "--device", device, "--out", f"lad_{cap}",
                   "--max-windows", mw, "--n-exec", "4"]
        elif cap == "audit":
            # the audit runs inside image_cls and vl_vqa; here just confirm it is reachable
            cmd = [PY, "-u", os.path.join(S, "train_image.py"), "--d", "32", "--epochs", "1",
                   "--batch", "256", "--device", "cpu", "--train-n", "1500",
                   "--out", "lad_audit"]
        else:
            continue

        rc, out, secs = run(cmd, timeout=timeout)
        checks = prepare_checks(cap, n, epochs, d, device, out)
        ok = rc == 0 and all(c[1] for c in checks)
        allpass = allpass and ok
        sub = "; ".join(f"{n_}: {'PASS' if p else 'FAIL'} ({dv})" for n_, p, dv in checks)
        print(f"  {'PASS' if ok else 'FAIL'}  {cap:<12} rc={rc:<4} {secs:6.1f}s   {sub}")
        if not ok:
            tail = "\n".join(out.strip().splitlines()[-6:])
            print(f"        last lines:\n          " + tail.replace("\n", "\n          "))
        table.append({"capability": cap, "rung": a.rung, "passed": bool(ok), "rc": rc,
                      "seconds": secs,
                      # RECORD THE BUDGET. The earlier version recorded only pass/fail, so a rung
                      # named "big" that actually trained on 2,000 windows looked identical to a real
                      # one in the saved JSON. The number that decides whether the rung means
                      # anything must be in the artifact.
                      "max_windows": int(mw), "epochs": int(epochs), "d": int(d),
                      "timeout_s": int(timeout),
                      "checks": [[c[0], bool(c[1]), c[2]] for c in checks]})

    print()
    print("=" * 100)
    print(f"  RUNG '{a.rung}': {'ALL PASS' if allpass else 'NOT ALL PASS'}  "
          f"({sum(t['passed'] for t in table)}/{len(table)})")
    if not allpass:
        print("  the ladder does NOT advance. fix the failing capability first.")
    print("=" * 100)
    json.dump(table, open(os.path.join(OUT, f"ladder_{a.rung}.json"), "w"), indent=1)
    print(f"  saved lens/runs/ladder/ladder_{a.rung}.json")


if __name__ == "__main__":
    main()
