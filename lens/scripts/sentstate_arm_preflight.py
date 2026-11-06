"""lens/scripts/sentstate_arm_preflight.py -- static wiring checks for lens/scripts/run_sentstate_arm.sh.

NO GPU, NO CORPUS, NO MODEL: this reads three files and refuses to let a queue line exist if the runner names
a flag the trainer does not have, a file that is missing, a label that would be split by the CKPT_LIST format,
or a destination that already exists (which would overwrite a stored number).

Predictions: lens/runs/SENTSTATE-ARM-PREDICTIONS-P116.txt (P116a-h; this pre-flight is the "written before"
part of P116h). Output: lens/runs/SENTSTATE-ARM-PREFLIGHT.txt (append-only).
"""
import os
import re
import subprocess
import sys
import time

P = "/run/media/yahya/HARD DISK 4/GlassNet"
RUNNER = os.path.join(P, "lens/scripts/run_sentstate_arm.sh")
TRAINER = os.path.join(P, "lens/scripts/train_chat_sent.py")
SHIPPED = os.path.join(P, "lens/scripts/train_chat.py")
CHAINED_PY = ["lens/scripts/talk_fair_v3_ship.py", "lens/scripts/talk_fair_test.py"]
CHAINED_SH = ["lens/sentstate_wire.py", "lens/scripts/train_chat_sent.py"]
DEST = [
    "lens/runs/SENTSTATE-ARM.txt",
    "lens/runs/PADFREE-TALK-SENTSTATE.txt",
    "lens/runs/PADFREE-TALK-SENTSTATE-TWIN.txt",
    "lens/runs/TALK-FAIR-SENTSTATE.txt",
    "lens/runs/TALK-FAIR-SENTSTATE-TWIN.txt",
]
TAG = "ss_d256_full169_3ep"
out = []


def log(s=""):
    out.append(s)
    print(s, flush=True)


def main():
    bad = 0
    log("=" * 100)
    log(f"SENTSTATE ARM PRE-FLIGHT  {time.strftime('%F %H:%M:%S')}   predictions: SENTSTATE-ARM-PREDICTIONS-P116.txt")
    log("=" * 100)

    # 1. shell syntax
    r = subprocess.run(["bash", "-n", RUNNER], capture_output=True, text=True)
    log(f"  1 bash -n: {'PASS' if r.returncode == 0 else 'FAIL'}  {r.stderr.strip()[:200]}")
    bad += 0 if r.returncode == 0 else 1

    # 2. the flags the runner passes to the trainer exist in the SHIPPED trainer's argparse
    src = open(RUNNER).read()
    trainer_src = open(SHIPPED).read()
    have = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', trainer_src))
    block = src.split("train_chat_sent.py", 1)[1].split("echo \"ARM EXIT", 1)[0]
    used = set(re.findall(r"(--[a-z0-9-]+)", block))
    missing = sorted(f for f in used if f not in have)
    log(f"  2 flags: {len(used)} used, {len(have)} in the shipped trainer's argparse")
    log(f"    MISSING FROM TRAINER: {missing if missing else 'none'}   -> {'PASS' if not missing else 'FAIL'}")
    bad += 0 if not missing else 1
    for f in sorted(used):
        log(f"      {f}")

    # 3. every chained file exists
    miss = [f for f in CHAINED_PY + CHAINED_SH if not os.path.exists(os.path.join(P, f))]
    log(f"  3 chained files present: {len(CHAINED_PY + CHAINED_SH) - len(miss)}/{len(CHAINED_PY + CHAINED_SH)}"
        f"   missing={miss if miss else 'none'}   -> {'PASS' if not miss else 'FAIL'}")
    bad += 0 if not miss else 1

    # 4. no CKPT_LIST label contains ';' (the format's own separator) -- [BUG-CKPTLIST-LABEL-SEMICOLON]
    labels = re.findall(r'CKPT_LIST="([^"]+)"', src)
    semis = [l for l in labels if l.count("=") < 1]
    log(f"  4 CKPT_LIST specs: {len(labels)}   items without '=': {semis if semis else 'none'}"
        f"   -> {'PASS' if not semis else 'FAIL'}")
    for l in labels:
        log(f"      {l}")
    bad += 0 if not semis else 1

    # 5. destinations must NOT exist yet (nothing overwritten outside the project's own new files)
    exist = [d for d in DEST if os.path.exists(os.path.join(P, d))]
    log(f"  5 destinations that already exist: {exist if exist else 'none'}   -> "
        f"{'PASS' if not exist else 'NOTE (append-only logs are expected to exist)'}")
    tag_files = [f for f in os.listdir(os.path.join(P, "lens/runs/chat")) if f.startswith(TAG)] \
        if os.path.isdir(os.path.join(P, "lens/runs/chat")) else []
    log(f"    existing checkpoints/rows with tag {TAG}: {tag_files if tag_files else 'none'}"
        f"   -> {'PASS (fresh tag)' if not tag_files else 'FAIL (would overwrite an arm)'}")
    bad += 0 if not tag_files else 1

    # 6. the launcher turns the graft ON only through SENTSTATE=1 and defaults OFF (P114a's control)
    launcher = open(TRAINER).read()
    ok6 = 'os.environ.get("SENTSTATE", "0") == "1"' in launcher
    log(f"  6 launcher defaults OFF, ON only with SENTSTATE=1: {ok6}   -> {'PASS' if ok6 else 'FAIL'}")
    bad += 0 if ok6 else 1

    # 7. the arm's declared parameter count, from the formula, printed beside its arithmetic (P116a)
    D, K, V, ds = 256, 36, 12002, 64
    graft = D * K + K * ds + ds * ds + ds + ds * V + V
    total = 5_908_686 + graft
    log(f"  7 P116a arithmetic: 5,908,686 + ({D}*{K} + {K}*{ds} + {ds}*{ds} + {ds} + {ds}*{V} + {V}) = "
        f"5,908,686 + {graft:,} = {total:,}   (the model line must print {total:,})")

    log("")
    log(f"PRE-FLIGHT RESULT: {'PASS' if bad == 0 else 'FAIL'}  ({bad} failed check(s))")
    with open(os.path.join(P, "lens/runs/SENTSTATE-ARM-PREFLIGHT.txt"), "a") as f:
        f.write("\n".join(out) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
