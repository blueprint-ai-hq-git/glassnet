#!/bin/bash
# unk_drop_movecheck.sh — verify the --drop-unk block was MOVED BELOW the window cap without changing anything
# it does (2026-10-24 self-wake). Two claims, both measured:
#   (1) THE WEIGHTS ARE UNCHANGED: the d=64 drop-unk arm's epoch line is digit-identical to the one measured
#       before the move (lens/runs/_tmp/unkpf_dropunk.out, attempt 3 of the pre-flight).
#   (2) THE PRINT NOW DESCRIBES THE SET THAT IS TRAINED ON: the dropped count/percent now come from the CAPPED
#       window set, so the count can be compared against the arm's own training set (before: 12,549 of 300,000
#       built, while the arm trained on 60,000).
# CPU only, cores 4,5, nice 19 (the GPU is holding the ours-method sweep). Writes ONLY
# lens/runs/UNK-DROP-MOVECHECK.txt and the new --out tag unkpf_movecheck_*.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export PYTHONPATH="$PWD/lens/scripts"
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
OUT=lens/runs/UNK-DROP-MOVECHECK.txt
PREV=lens/runs/_tmp/unkpf_dropunk.out
NEW=lens/runs/_tmp/unkpf_movecheck.out
[ -f "$PREV" ] || { echo "PREVIOUS RUN MISSING ($PREV) — claim (1) cannot be checked; refusing to report"; exit 2; }
{
echo "=== DROP-UNK MOVE CHECK $(date '+%F %H:%M:%S') ==="
echo "before: the block sat in the assistant-mask section (above the window cap) -> $PREV"
echo "after:  the block sits just below the window cap -> this run"
nice -n 19 ionice -c3 taskset -c 4,5 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 64 --window 32 --epochs 1 --batch 128 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 \
  --seed 0 --device cpu --assistant-only --train-turn-end --direct --direct-mode tied \
  --max-windows 60000 --method standard --drop-unk --out unkpf_movecheck > "$NEW" 2>&1
echo "trainer exit=$?"
echo
echo "--- the flag's own printed lines, BEFORE the move (from $PREV) ---"
grep "drop-unk:" "$PREV"
echo "--- the flag's own printed lines, AFTER the move ---"
grep "drop-unk:" "$NEW"
echo
.venv-gpu/bin/python3 - "$PREV" "$NEW" <<'PY'
import re, sys
def ep(path):
    for ln in open(path):
        if re.match(r"\s*ep\s+\d+/\d+", ln) and "loss" in ln:
            return ln.strip()
    return None

# THE TIMING FIELD IS NOT PART OF THE RESULT. The first version of this check compared the WHOLE line, so it
# printed "FAIL — THE MOVE CHANGED THE RUN" on a pair whose only difference was the wall clock, "(77s)" against
# "(76s)" — a verdict read off the wrong thing, which is this project's signature bug class. The trailing
# "(NNNs)" is stripped, and the two full lines are still printed so nothing is hidden by the strip.
def strip_time(s):
    return re.sub(r"\s*\(\d+s\)\s*$", "", s) if s else s

a, b = ep(sys.argv[1]), ep(sys.argv[2])
sa, sb = strip_time(a), strip_time(b)
print("CLAIM 1 — THE RESULT IS UNCHANGED (the whole epoch line except the wall-clock field, digit for digit)")
print("  before the move: " + str(a))
print("  after  the move: " + str(b))
print("  compared after stripping the (NNNs) timing field, which is not a measurement of the model:")
if sa is not None and sa == sb:
    print("  -> PASS (the two lines are identical field for field; the timing field was the only difference)")
else:
    print("  -> FAIL — THE MOVE CHANGED THE RUN; REVERT IT, do not explain it")
    print("     before: " + str(sa))
    print("     after : " + str(sb))

def drop(path):
    for ln in open(path):
        if "drop-unk:" in ln and "removed from the loss" in ln:
            m = re.search(r"drop-unk: ([\d,]+) training targets", ln)
            p = re.search(r"([\d.]+)% of the ([\d,]+) training targets this run actually trains on", ln)
            w = re.search(r"([\d.]+)% of the WEIGHTED", ln)
            return (int(m.group(1).replace(",", "")) if m else None,
                    (float(p.group(1)), int(p.group(2).replace(",", ""))) if p else None,
                    float(w.group(1)) if w else None)
    return (None, None, None)
da, db = drop(sys.argv[1]), drop(sys.argv[2])
print()
print("CLAIM 2 — THE PRINT NOW DESCRIBES THE TRAINING SET")
print(f"  before: dropped {da[0]:,}  -> {da[1]}  ({da[2]}% of weighted)   [denominator = BUILT set]")
print(f"  after : dropped {db[0]:,}  -> {db[1]}  ({db[2]}% of weighted)   [denominator = the CAPPED set]")
if da[2] is not None and db[2] is not None:
    print(f"  the RATE is what must not change: {da[2]}% -> {db[2]}%  -> "
          f"{'PASS (same rate, spot on)' if abs(da[2]-db[2]) < 0.15 else 'CHECK: rate moved'}")
print("  the count itself MUST change, because it is now a count over 60,000 windows instead of 300,000")
PY
echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> "$OUT" 2>&1
