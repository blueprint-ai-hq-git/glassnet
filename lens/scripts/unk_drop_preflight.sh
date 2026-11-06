#!/bin/bash
# unk_drop_preflight.sh — CPU WIRING pre-flight for the new --drop-unk flag (predictions P89a-c,
# lens/runs/UNK-DROP-PREDICTIONS-P89.txt, written before the trainer was edited).
# THIS IS A WIRING TEST, NOT A QUALITY CLAIM: d=64 / 60,000 windows / 1 epoch. Nothing here may be
# reported as a result about the model.
# It runs THREE tiny arms so the default path can be proven unchanged:
#   1. the PRE-CHANGE copy of the trainer (lens/runs/_tmp/train_chat_prefix_unkdrop_20261024-075441.py)
#   2. the new trainer, --drop-unk ABSENT      -> P89a: its epoch line must be digit-identical to (1)
#   3. the new trainer, --drop-unk ON          -> P89b/P89c: the cut is real, non-zero and counted
# CPU ONLY on cores 4,5 (the GPU holds the ours-method settings sweep; this must not disturb it).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
# The PRE-CHANGE copy lives in lens/runs/_tmp/, so `import chat_concepts` / `import chat_data` (which the
# real trainer finds because python prepends the SCRIPT's own directory) is NOT importable from there.
# FIRST ATTEMPT FAILED EXACTLY SO (ModuleNotFoundError: chat_concepts, logged in
# lens/runs/UNK-DROP-PREFLIGHT.txt at 07:56) -- this is the fix, and for lens/scripts/train_chat.py the
# setting is a no-op because python already prepends that same directory.
export PYTHONPATH="$PWD/lens/scripts"
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"

OUT=lens/runs/UNK-DROP-PREFLIGHT.txt
TMP=lens/runs/_tmp
PRE=$TMP/train_chat_prefix_unkdrop_20261024-075441.py
[ -f "$PRE" ] || { echo "PRE-CHANGE COPY MISSING: $PRE — refusing to run (P89a is unanswerable without it)"; exit 2; }

COMMON="--d 64 --window 32 --epochs 1 --batch 128 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 \
--seed 0 --device cpu --assistant-only --train-turn-end --direct --direct-mode tied \
--max-windows 60000 --method standard"

{
echo "================================================================================================"
echo "UNK-DROP PRE-FLIGHT (P89a-c)  $(date '+%F %H:%M:%S')   d=64, 60,000 windows, 1 epoch, CPU, cores 4,5"
echo "predictions: lens/runs/UNK-DROP-PREDICTIONS-P89.txt   (written before the trainer edit)"
echo "this is a WIRING test; d=64 quality is meaningless and is not reported as anything else"
echo "================================================================================================"
} >> "$OUT"

run () {  # $1 = tag, $2 = python file, $3 = extra flags
  # BUG FIXED HERE (2026-10-24, MEASURED): the first version declared the filename on the SAME `local` line as
  # the tag -- `local tag="$1" py="$2" extra="$3" f="$TMP/unkpf_${tag}.out"` -- and on this machine's bash
  # (GNU bash 5.3.9) that expands ${tag} to the EMPTY string, so all three arms wrote ONE file called
  # unkpf_.out and every `cat` dumped the accumulated contents into the log again (the duplicated blocks in
  # lens/runs/UNK-DROP-PREFLIGHT.txt). Reproduced in isolation, both forms side by side:
  #   A: local tag="$1" ... f="$TMP/unkpf_${tag}.out"  -> f=[.../unkpf_.out]            (WRONG)
  #   B: local tag="$1" ; local f="$TMP/unkpf_${tag}.out" -> f=[.../unkpf_prechange.out] (right)
  # (/home/yahya/.hermes/cache/scratch/tagtest.sh; the observed symptom was the first attempt's real failure
  # mode reading "No such file or directory: 'lens/runs/_tmp/unkpf_prechange.out'" in the grading step.)
  local tag="$1" py="$2" extra="$3"
  local f="$TMP/unkpf_${tag}.out"
  {
  echo "------------------------------------------------------------------------------------------------"
  echo "ARM $tag   interpreter=$py   extra='${extra:-<none>}'"
  echo "------------------------------------------------------------------------------------------------"
  } >> "$f"
  taskset -c 4,5 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u "$py" $COMMON $extra --out "unkpf_${tag}" >> "$f" 2>&1
  echo "exit=$?" >> "$f"
  cat "$f" >> "$OUT"
}

run prechange   "$PRE" ""
run newdefault  lens/scripts/train_chat.py ""
run dropunk     lens/scripts/train_chat.py "--drop-unk"

echo "" >> "$OUT"
echo "=== GRADING (P89a-c) ===" >> "$OUT"
.venv-gpu/bin/python3 - "$TMP" "$OUT" <<'PY' >> "$OUT" 2>&1
import os, re, sys
tmp, out = sys.argv[1], sys.argv[2]

def read(tag):
    return open(os.path.join(tmp, f"unkpf_{tag}.out")).read()

def epoch_line(txt):
    for ln in txt.splitlines():
        if re.match(r"\s*ep\s+\d+/\d+", ln) and "loss" in ln:
            return ln.strip()
    return None

def exits(txt):
    return [int(m.group(1)) for m in re.finditer(r"^exit=(\d+)", txt, re.M)]

def drop_lines(txt):
    return [l.strip() for l in txt.splitlines() if "drop-unk:" in l]

pre, new, dr = read("prechange"), read("newdefault"), read("dropunk")
ep_pre, ep_new, ep_dr = epoch_line(pre), epoch_line(new), epoch_line(dr)
ex_pre, ex_new, ex_dr = exits(pre), exits(new), exits(dr)

print()
print("P89a THE DEFAULT PATH IS UNCHANGED (new file, --drop-unk ABSENT, against the pre-change copy)")
print(f"  pre-change epoch line: {ep_pre}")
print(f"  new file   epoch line: {ep_new}")
print(f"  exits: pre={ex_pre} new={ex_new}")
same = (ep_pre is not None and ep_pre == ep_new)
newlines = [l for l in new.splitlines() if "drop-unk" in l]
print(f"  new lines mentioning drop-unk in the DEFAULT run: {len(newlines)} (predicted 0)")
print(f"  -> P89a {'PASS' if (same and not newlines and ex_new == [0]) else 'FAIL'}"
      f"   (digit-identical epoch line: {same}; clean exit: {ex_new == [0]})")
if not same:
    print("  NOTE: the edit changed the default path. It is to be REVERTED, not explained (P89a).")

print()
print("P89b THE FLAG IS WIRED AND NOT A NO-OP")
dl = drop_lines(dr)
for l in dl:
    print("  trainer: " + l)
m = re.search(r"drop-unk: ([\d,]+) training targets .*?-- ([\d.]+)% of all training targets, ([\d.]+)% of the WEIGHTED",
              dr)
tr_dropped = int(m.group(1).replace(",", "")) if m else None
tr_all_pct, tr_w_pct = (float(m.group(2)), float(m.group(3))) if m else (None, None)
print(f"  trainer's dropped count: {tr_dropped}   ({tr_all_pct}% of all targets, {tr_w_pct}% of weighted)")
print(f"  dropunk run epoch line: {ep_dr}")
print(f"  exits: prechange={ex_pre} newdefault={ex_new} dropunk={ex_dr}")
ok_b = (tr_dropped is not None and tr_dropped > 0 and ex_dr == [0])
print(f"  -> P89b {'PASS' if ok_b else 'FAIL'}  (non-zero count: {bool(tr_dropped)}, clean exit: {ex_dr == [0]})")
if ep_dr == ep_new:
    print("  WARNING: the drop-unk arm's epoch line is IDENTICAL to the default arm's, which would mean"
          " the flag changed nothing that matters. Recorded as a wiring failure.")

print()
print("P89c THE SIZE OF THE CUT, INDEPENDENTLY (same window set, rebuilt here, no mask replica)")
print("  NOTE ON WHAT IS INDEPENDENT: the share below is recomputed from the data by this script"
      " (unweighted, over the same seeded 60,000-window cap the trainer printed). The WEIGHTED share is")
print("  the trainer's own count; this script checks that the two agree where they must (the trainer's")
print("  '% of all training targets' is exactly the same quantity as the independent one).")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "lens/scripts"))
import chat_data                   # noqa: E402
import numpy as np                  # noqa: E402
d = chat_data.build()
X, Y, S = chat_data.make_windows(d["ids"], d["bounds"], d["split"], 32, which=0,
                                max_windows=max(60000 * 2, 300000), return_index=True)
keep = np.random.default_rng(0).choice(len(X), size=60000, replace=False)
keep.sort()
Yc = Y[keep]
ind_all_pct = float((Yc == 0).mean() * 100)
print(f"  windows rebuilt: {len(X):,}, capped to {len(Yc):,} with seed 0 (the trainer printed its own cap above)")
print(f"  independent all-target <unk> share : {ind_all_pct:.3f}%")
print(f"  trainer's  all-target <unk> share : {tr_all_pct}%")
agree = (tr_all_pct is not None and abs(ind_all_pct - tr_all_pct) < 0.005)
print(f"  -> P89c agreement: {'AGREE' if agree else 'DISAGREE'}  (bar 0.005 points)")
print(f"  -> P89c band check (predicted 4-9% of WEIGHTED targets dropped): "
      f"{'INSIDE' if (tr_w_pct is not None and 4.0 <= tr_w_pct <= 9.0) else 'OUTSIDE'}  "
      f"(measured {tr_w_pct}%)")
PY
echo "=== DONE $(date '+%F %H:%M:%S') ===" >> "$OUT"
