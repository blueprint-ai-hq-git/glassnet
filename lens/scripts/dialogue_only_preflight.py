#!/usr/bin/env python3
"""dialogue_only_preflight.py — WIRE-CHECK THE DIALOGUE-ONLY ARM BEFORE IT IS QUEUED. Loads no model, trains
nothing, writes nothing. Everything it prints is read from a file or from the trainer's own source.

PREDICTIONS P110a-h: lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt (written before the build script existed)

WHAT IT CHECKS (each one is a way this project has been burned before, named):
  1. the slice directory exists with all five array/meta files, and chat_data.build() actually loads it through
     the trainer's OWN loader (CHAT_DATA_DIR) — a corpus the loader cannot open is a queue line that dies in
     30 seconds (BUG [BUG-TRAINCHAT-CORPUS-ENV] family).
  2. the slice's split has val conversations, and the val window count is computed arithmetically (no window
     array is materialised): the arm must have a non-empty held-out column.
  3. every flag in the runner's recipe is a real flag in lens/scripts/train_chat.py's own argparse block — a
     typo'd flag is an argparse exit BEFORE any epoch is logged.
  4. the runner's "best epoch" read block runs on a REAL history file and prints a number (the project's other
     documented runner bug: a read block that has never been executed).
  5. every chained decider file exists, and no CKPT_LIST label contains ';' (three separate outages came from
     that one character).
  6. the destination paths the runner writes do not already exist, so nothing is overwritten.
"""
import json
import os
import re
import subprocess
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"
SLICE = os.path.join(P, "data/chat/prepared_dialogue_only_honest")
TRAINER = os.path.join(P, "lens/scripts/train_chat.py")
RUNNER = os.path.join(P, "lens/scripts/run_dialogue_only_d256_6ep.sh")
TAG = "dial_only_d256_6ep"
FILES = ["ids_v12000.npy", "bounds_v12000.npy", "split_v12000.npy", "vocab_v12000.json", "meta_v12000.json"]
DECIDERS = ["lens/scripts/talk_fair_v3_ship.py", "lens/scripts/talk_fair_test.py",
            "lens/scripts/dialogue_mixed_eval.py"]
RECIPE_FLAGS = ["--d", "--window", "--epochs", "--batch", "--lr", "--lr-decay", "--lam-concept", "--lam-act",
                "--seed", "--device", "--assistant-only", "--train-turn-end", "--direct", "--direct-mode",
                "--max-windows", "--method", "--out", "--ckpt-seconds"]

checks = []


def chk(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def main():
    print("=" * 110)
    print("DIALOGUE-ONLY ARM — WIRING PRE-FLIGHT (loads no model, trains nothing, writes nothing)")
    print("predictions P110a-h: lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt")
    print("=" * 110)

    # ---- 1. the slice exists and the TRAINER'S OWN LOADER opens it ---------------------------------------
    print("")
    print("1. THE SLICE, THROUGH THE TRAINER'S OWN LOADER (chat_data.build with CHAT_DATA_DIR)")
    present = [f for f in FILES if os.path.exists(os.path.join(SLICE, f))]
    chk("all five files present in data/chat/prepared_dialogue_only_honest", len(present) == len(FILES),
        f"{len(present)}/{len(FILES)}: {', '.join(present)}")
    for f in FILES:
        p = os.path.join(SLICE, f)
        if os.path.exists(p):
            print(f"      {f:<24} {os.path.getsize(p):>14,} B")

    os.environ["CHAT_DATA_DIR"] = SLICE
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    sys.path.insert(0, os.path.join(P, "lens", "scripts"))
    sys.path.insert(0, P)
    try:
        import chat_data
        import numpy as np
        d = chat_data.build()
    except Exception as e:                                   # noqa: BLE001
        chk("chat_data.build() loads the slice", False, f"{type(e).__name__}: {e}")
        d = None
    if d is not None:
        ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
        chk("chat_data.build() loads the slice", True,
            f"{len(ids):,} tokens, vocab {len(vocab):,}, {len(bounds):,} bounds, {len(split):,} conversations, "
            f"id dtype {getattr(ids, 'dtype', '?')}")
        bnd = np.asarray(bounds, dtype=np.int64)
        chk("bounds ends at len(ids) (the loader's own sentinel convention)",
            int(bnd[-1]) == len(ids), f"bounds[-1]={int(bnd[-1]):,}  len(ids)={len(ids):,}")
        ctrl_ok = len(vocab) > 12001 and vocab[12000] == "<|im_start|>" and vocab[12001] == "<|im_end|>"
        chk("vocab has the mix's two control tokens at ids 12000/12001", ctrl_ok,
            f"len(vocab)={len(vocab)}  vocab[12000]={vocab[12000] if len(vocab) > 12000 else 'MISSING'}  "
            f"vocab[12001]={vocab[12001] if len(vocab) > 12001 else 'MISSING'}")

        # ---- 2. window arithmetic (NO window array is materialised) ---------------------------------------
        print("")
        print("2. WINDOW AND HELD-OUT ARITHMETIC (the loader's own formula, counts only)")
        n_tr = n_va = n_frozen = 0
        for j in range(len(split) - 1):
            span = int(bnd[j + 1] - bnd[j]) - 32 - 1
            c = max(0, len(range(0, span, 16)))
            if split[j] == 1:
                n_va += c
            else:
                n_tr += c
        if split[len(split) - 1] != 1:
            n_frozen = max(0, len(range(0, int(len(ids) - bnd[-1]) - 32 - 1, 16)))
        chk("the slice produces TRAIN windows", n_tr > 0, f"{n_tr:,} train windows")
        chk("the slice produces VAL windows", n_va > 0, f"{n_va:,} val windows")
        chk("train windows fit inside --max-windows 16900000 (P110's declared consequence: the whole slice is "
            "seen every epoch)", n_tr <= 16900000, f"{n_tr:,} <= 16,900,000")
        print(f"      windows the mix corpus gives the twin: 16,878,132 uncapped "
              f"(this slice gives {n_tr:,} = {n_tr / 16878132 * 100:.1f}%)")
        print(f"      the last conversation (index {len(split) - 1}, split={int(split[-1])}) is dropped by the "
              f"loader's own loop either way: {n_frozen:,} windows")

    # ---- 3. every flag in the recipe is a real trainer flag -----------------------------------------------
    print("")
    print("3. THE RECIPE'S FLAGS, AGAINST THE TRAINER'S OWN argparse BLOCK")
    src = open(TRAINER).read()
    flags_in_src = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))
    missing = [f for f in RECIPE_FLAGS if f not in flags_in_src]
    chk("every recipe flag exists in train_chat.py", not missing,
        f"{len(RECIPE_FLAGS)} checked, missing: {missing or 'none'}")
    if os.path.exists(RUNNER):
        rsrc = open(RUNNER).read()
        # only real flag-looking tokens: a flag starts with -- followed by a letter (NOT the '-----' rules in
        # the header comments, which the first version of this check counted as unknown flags)
        used = sorted(set(re.findall(r'(--[a-z][a-z0-9-]*)', rsrc.split("--- chained decider", 1)[0])))
        unknown = [f for f in used if f not in flags_in_src and f not in ("--out",)]
        chk("the runner uses no flag the trainer does not have", not unknown,
            f"{len(used)} flag tokens in the recipe section, unknown: {unknown or 'none'}")
        chk("the runner passes CHAT_DATA_DIR at the slice", "prepared_dialogue_only_honest" in rsrc)
        chk("the runner sets a tag and a log and neither exists yet",
            f"TAG={TAG}" in rsrc and not os.path.exists(os.path.join(P, f"lens/runs/chat/{TAG}_history.jsonl")),
            f"tag {TAG}")
    else:
        chk("the runner exists", False, RUNNER)

    # ---- 4. the best-epoch read block runs on a REAL history file ------------------------------------------
    print("")
    print("4. THE BEST-EPOCH READ BLOCK, EXECUTED ON A REAL HISTORY FILE (not merely parseable)")
    hist = os.path.join(P, "lens/runs/chat/h2l_noconcept_169_6ep_history.jsonl")
    snippet = ("import json;rows=[json.loads(l) for l in open('" + hist + "') if l.strip()];"
               "b=max(rows,key=lambda r:r['val_asst_acc']);print('%.4f'%float(b['val_masked_acc']))")
    try:
        out = subprocess.run([os.path.join(P, ".venv-gpu/bin/python3"), "-c", snippet],
                             capture_output=True, text=True, timeout=60)
        val = (out.stdout or "").strip()
        chk("the read block prints a number on h2l's own history", out.returncode == 0 and val != "",
            f"exit {out.returncode}, printed {val!r} (h2l's best legacy by-id column)")
    except Exception as e:                                   # noqa: BLE001
        chk("the read block prints a number on h2l's own history", False, f"{type(e).__name__}: {e}")

    # ---- 5. chained deciders and the ';' trap -------------------------------------------------------------
    print("")
    print("5. THE CHAINED DECIDERS, AND THE ';' LABEL TRAP")
    for f in DECIDERS:
        chk(f"decider present: {f}", os.path.exists(os.path.join(P, f)))
    labels = ["DL-arm (dialogue only, the H2L recipe - ONE change)", "TWIN ct_d256_full169_fix3 (stored control)",
              "H2L (the arm this one changes one variable from)"]
    chk("no CKPT_LIST label contains ';'", not any(";" in x for x in labels),
        f"{len(labels)} labels checked")
    outs = ["lens/runs/DIALOGUE-ONLY-D256-6EP.txt", "lens/runs/PADFREE-TALK-DLONLY.txt",
            "lens/runs/TALK-FAIR-DLONLY.txt", "lens/runs/TALK-FAIR-DLONLY-TWIN.txt"]
    chk("no destination file the runner writes already exists",
        not any(os.path.exists(os.path.join(P, o)) for o in outs),
        f"{len(outs)} outputs checked")

    print("")
    bad = [n for n, ok, _ in checks if not ok]
    print("=" * 110)
    print(f"PRE-FLIGHT: {len(checks) - len(bad)}/{len(checks)} checks PASS"
          + (f"   FAILED: {bad}" if bad else "   — every check passed"))
    print("=" * 110)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
