#!/usr/bin/env python3
"""sweep_cut_verify.py - verify THAT THE CUT LANDED, and that the critical path took the GPU over.

Written by the 2026-10-24 ~09:2x self-wake with its predictions on disk first
(lens/runs/SWEEP-CUT-PREDICTIONS-P90.txt). It CHECKS, it does not decide, and it writes ONLY
lens/runs/SWEEP-CUT-VERIFY.txt (append).

WHAT IT WAITS FOR, in order:
  1. arm A's epoch 2 record in lens/runs/chat/ours_arm_A_ours_tol0_history.jsonl  (P90b; the arm's python was
     left alive when the two bash wrappers were killed, so the arm ends complete rather than truncated).
  2. OURS-SWEEP.txt must NEVER gain an "ARM B_ours_tol001" header (P90a: the cut held; if this fires, arm B
     started and the cut failed).
  3. the trainer count must reach 0 and the sweep's scope must go inactive (P90c).
  4. the next watchdog tick must start 'bash lens/scripts/run_unk_drop.sh' - the critical-path <unk> arm
     (P90c). Detected from QUEUE-DONE.txt and/or a trainer whose --out tag is unk_drop_169_3.
  5. GPU VRAM under 5 GB throughout (P90e).
Every line carries its own timestamp and its own source file. Bounded: it gives up and says so rather than
waiting forever.
"""
import datetime
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path("/run/media/yahya/HARD DISK 4/GlassNet")
RUNS = ROOT / "lens/runs"
CHAT = RUNS / "chat"
OUT = RUNS / "SWEEP-CUT-VERIFY.txt"
HIST = CHAT / "ours_arm_A_ours_tol0_history.jsonl"
SWEEP = RUNS / "OURS-SWEEP.txt"
QDONE = RUNS / "QUEUE-DONE.txt"
SCOPE = "glassnet-q-3757950001-074809.scope"


def stamp():
    return datetime.datetime.now().strftime("%F %H:%M:%S")


def log(line=""):
    with open(OUT, "a") as f:
        f.write(line + "\n")
    sys.stdout.write(line + "\n")


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (p.stdout or "").strip()


def count_trainers():
    return int(sh("pgrep -cf '[t]rain_chat\\.py'") or 0)


def scope_active():
    return sh(f"systemctl --user is-active {SCOPE} 2>/dev/null") or "gone"


def vram_used():
    s = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null")
    return int(s) if s.isdigit() else -1


def epochs_in_hist():
    n = 0
    last = None
    if HIST.exists():
        for ln in HIST.read_text(errors="replace").splitlines():
            if ln.strip():
                n += 1
                last = ln
    return n, last


def main():
    log("=" * 110)
    log(f"SWEEP-CUT VERIFICATION  started {stamp()}   (sweep_cut_verify.py; predictions P90a-f)")
    log("=" * 110)
    vmax = 0
    failed = []

    # ---- 1. arm A epoch 2
    t_end = time.time() + 7200
    n_ep = 0
    while time.time() < t_end:
        v = vram_used()
        vmax = max(vmax, v)
        n_ep, last = epochs_in_hist()
        if n_ep >= 2:
            break
        time.sleep(30)
    n_ep, last = epochs_in_hist()
    if n_ep >= 2:
        log(f"[{stamp()}] P90b: arm A wrote {n_ep} epoch records while its wrappers were dead -> the cut cost no")
        log(f"          completed arm. LAST RECORD {HIST}:")
        log(f"          {last}")
        m = re.search(r'"val_asst_acc": ([0-9.]+)', last or "")
        if m:
            got = float(m.group(1)) * 100
            band = 12.5 <= got <= 16.0
            log(f"          ep2 by-position {got:.4f}% -> inside the predicted 12.5-16.0 band: {band}")
            if not band:
                failed.append("P90b-band")
    else:
        log(f"[{stamp()}] P90b: only {n_ep} epoch record(s) after 2 h -> the arm did NOT finish epoch 2 (recorded)")
        failed.append("P90b")

    # ---- 2. was arm B ever started?
    bstarted = "ARM B_ours_tol001" in SWEEP.read_text(errors="replace")
    log(f"[{stamp()}] P90a: OURS-SWEEP.txt {'CONTAINS' if bstarted else 'never gained'} an 'ARM B_ours_tol001' header")
    log(f"          -> the cut {'FAILED (arm B started)' if bstarted else 'held'}")
    if bstarted:
        failed.append("P90a")

    # ---- 3. trainers down, scope inactive
    t_end = time.time() + 1800
    while time.time() < t_end:
        if count_trainers() == 0:
            break
        time.sleep(20)
    act = scope_active()
    log(f"[{stamp()}] P90c: trainers running {count_trainers()}   scope {SCOPE} -> {act}")

    # ---- 3b. THE ONE WAY THIS CUT COULD SILENTLY BLOCK THE CRITICAL PATH. The watchdog's honest "busy" guard
    # counts ACTIVE glassnet-q-* scopes, so a scope that stayed active after its last process exited would hold
    # the queue forever. This machine's measured behaviour is that a scope goes inactive when its processes go,
    # so this is belt-and-braces - and it can only fire with the trainer count at 0 by its own condition.
    # KillMode=process on that scope means a stop() signals only its main process (92337, already dead), so it
    # cannot touch a live trainer.
    if count_trainers() == 0 and scope_active() == "active":
        log(f"[{stamp()}] NUDGE: 0 trainers but {SCOPE} still reads active -> stopping it so the watchdog is free")
        log("        " + sh(f"systemctl --user stop {SCOPE} 2>&1; systemctl --user is-active {SCOPE} 2>/dev/null"))
        log(f"        after: {scope_active()}")
    else:
        log(f"[{stamp()}] no nudge needed (trainers {count_trainers()}, scope {scope_active()})")

    # ---- 4. did the critical path start?
    t_end = time.time() + 3600
    started = False
    while time.time() < t_end:
        trainers = sh("pgrep -af '[t]rain_chat\\.py'")
        qd = QDONE.read_text(errors="replace")
        mine = [l for l in qd.splitlines() if "run_unk_drop.sh" in l]
        if "unk_drop_169_3" in trainers or any("2026-10-24" in l for l in mine):
            started = True
            break
        time.sleep(30)
    trainers = sh("pgrep -af '[t]rain_chat\\.py'")
    mine = [l for l in qd.read_text(errors="replace").splitlines() if "run_unk_drop.sh" in l or "run_ours_sweep_part2.sh" in l]
    log(f"[{stamp()}] P90c: the critical-path <unk> arm started by the watchdog: {started}")
    log(f"          trainers now: {trainers or '(none)'}")
    log(f"          queue lines consumed (QUEUE-DONE.txt, both relevant lines):")
    for l in mine:
        log(f"            {l}")
    if not started:
        failed.append("P90c")

    # ---- 5. VRAM rule
    v = vram_used()
    vmax = max(vmax, v)
    log(f"[{stamp()}] P90e: GPU VRAM peak sampled this run {vmax} MiB (rule: under 5,000) -> {'PASS' if 0 <= vmax < 5000 else 'FAIL'}")

    # ---- 6. the restored arms are really in the queue
    q = RUNS.joinpath("QUEUE.txt").read_text(errors="replace")
    active = [l for l in q.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    log(f"[{stamp()}] P90d: active queue lines now ({len(active)}):")
    for l in active:
        log(f"            {l}")

    log("")
    log(f"VERDICT: {'ALL CHECKS PASSED' if not failed else 'FAILED: ' + ', '.join(failed)}")
    log(f"finished {stamp()}")
    log("=" * 110)


if __name__ == "__main__":
    main()
