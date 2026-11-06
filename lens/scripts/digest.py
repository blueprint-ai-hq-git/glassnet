#!/usr/bin/env python3
"""digest.py - compact, accurate status text for the user's chat.

Reads the real result files, takes the LAST arms that actually completed, and prints for each one:
the honest number first, the raw number beside it, and the text verdict. Nothing is estimated; an arm
without a finished eval is shown as "still training".
"""
import glob
import os
import re
import subprocess

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
SRC = [f"{ROOT}/lens/runs/SMALL-BIG.txt", f"{ROOT}/lens/runs/H-BLOCK.txt",
       f"{ROOT}/lens/runs/HONEST-LADDER.txt"]

def arms():
    out = []
    for f in SRC:
        if not os.path.exists(f):
            continue
        blocks, cur = [], None
        for ln in open(f, errors="replace"):
            if ln.startswith("## ARM "):
                cur = {"header": ln.strip(), "lines": [], "file": os.path.basename(f)}
                blocks.append(cur)
            elif ln.startswith("####") and cur is not None:
                cur = None
            elif cur is not None:
                cur["lines"].append(ln.rstrip())
        out.extend(blocks)
    return out

def summarize(b):
    name = re.sub(r"^## ARM\s+", "", b["header"]).split()[0]
    masked = [l.strip() for l in b["lines"] if "assistant-masked" in l]
    accs = [float(re.search(r"acc ([0-9.]+)%", l).group(1)) for l in masked if re.search(r"acc ([0-9.]+)%", l)]
    verdict = next((l.strip() for l in b["lines"] if "VERDICT" in l), None)
    tq = next((l.strip() for l in b["lines"] if "this model's output" in l), None)
    final = next((l.strip() for l in b["lines"] if "final VAL" in l), None)
    if not accs:
        alive = subprocess.run(["pgrep", "-af", f"[t]rain_chat.*{name}"],
                               capture_output=True, text=True).stdout.strip()
        return f"  {name}: {'still training' if alive else 'waiting in the queue'} (no finished eval yet)"
    if len(accs) >= 2:
        num = f"honest {accs[1]:.2f}%  (raw {accs[0]:.2f}%)"
    else:
        num = f"{accs[0]:.2f}%  (single number, no raw/honest pair)"
    extra = ""
    if tq:
        m = re.search(r"(-[0-9.]+)", tq)
        if m:
            extra = f"   text {m.group(1)}"
    if verdict:
        extra += "  " + ("WORD SALAD" if "SALAD" in verdict else verdict.split(":")[-1].strip())
    return f"  {name}: {num}{extra}"

def running():
    try:
        ps = subprocess.run(["pgrep", "-af", "[t]rain_chat"], capture_output=True, text=True).stdout
    except Exception:
        ps = ""
    if not ps.strip():
        return "  nothing on the GPU"
    arm = re.search(r"--out ([A-Za-z0-9_]+)", ps)
    name = arm.group(1) if arm else "?"
    cks = sorted(glob.glob(f"{ROOT}/lens/runs/chat/{name}_ep*.npz"),
                 key=os.path.getmtime, reverse=True)
    ep = re.sub(r".*_ep", "", cks[0]).replace(".npz", "") if cks else "?"
    return f"  {name} at epoch {ep}"

def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"

def main():
    print("RUNNING NOW")
    print(running())
    print(f"  GPU {gpu()}")
    print()
    print("LATEST FINISHED ARMS (honest first, raw in brackets)")
    done = [summarize(b) for b in arms()]
    done = [d for d in done if "still training" not in d]
    for line in done[-4:]:
        print(line)
    print()
    print("STILL TRAINING")
    for line in [summarize(b) for b in arms()]:
        if "still training" in line:
            print(line)
    print()
    for label, pat in [("queue wrappers", "[r]un_tomorrow|[r]un_h_block|[r]un_corpus_ab|[r]un_full_schedule")]:
        ps = subprocess.run(["pgrep", "-af", pat], capture_output=True, text=True).stdout
        names = sorted({re.search(r"run_[a-z_]+", l).group(0) for l in ps.splitlines() if re.search(r"run_[a-z_]+", l)})
        print(f"{label}: {', '.join(names) if names else 'none'}")
    print(f"disk free: {subprocess.run(['df','-h',ROOT], capture_output=True, text=True).stdout.splitlines()[-1].split()[3]}")

if __name__ == "__main__":
    main()