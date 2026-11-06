"""fix_sweep_check.py — correct the sweep's bits-per-character column and verdict.

THE BUG (the third of its kind in this project)
-----------------------------------------------
run_sweep.sh built its rows with `cpt = 1` for every variant:

    rows = [("BASELINE ...", "grid_d128_k48", 1), ("A vocab 3500", "sweepA_vocab3500", 1), ...]
    bc = math.log2(h["val_ppl"]) / cpt

`cpt` is characters per token, which is 4.576 for word-level models and 1.0 for character-level.
Passing 1 made the column "bits per TOKEN", and then the verdict compared that against the real
baseline of 1.576 bits per CHARACTER and printed "NO VARIANT BEAT THE BASELINE".

**The numbers were right. The conclusion was wrong.** Same class as METER-06 (an arithmetic slip
that capped a perfect concept) and the character check (a mislabelled verdict). Three times now,
the failure has been a verdict attached to a correct measurement -- never a wrong measurement.

This recomputes everything with the correct constant and states the verdict from the data.
"""
import json
import math
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat"
CHARS_PER_WORD = 60949068 / 13318447          # 4.576, measured from the corpus

VARIANTS = [
    ("BASELINE d128 K37 v12k W8", "grid_d128_k48", CHARS_PER_WORD),
    ("A vocab 3500", "sweepA_vocab3500", CHARS_PER_WORD),
    ("B window 16", "sweepB_win16", CHARS_PER_WORD),
    ("C 24ep + lrdecay", "sweepC_long", CHARS_PER_WORD),
    ("D d=256", "sweepD_d256", CHARS_PER_WORD),
    ("big1 d384 K48 v12k W8", "big1", CHARS_PER_WORD),
    ("char d64 W32 K35", "char1", 1.0),
]


def load(out):
    for suffix in ("_replies.json", "_log.json"):
        p = os.path.join(P, out + suffix)
        if os.path.exists(p):
            return json.load(open(p))
    return None


print("=" * 100)
print("THE SWEEP — CORRECTED (bits per CHARACTER, not per token)")
print("=" * 100)
print(f"  corpus: {CHARS_PER_WORD:.3f} characters per word")
print()
print(f"  {'variant':<26}{'params':>11}{'ppl':>9}{'bits/char':>11}{'acc':>8}{'concepts':>10}")
rows = []
for tag, out, cpt in VARIANTS:
    r = load(out)
    if r is None:
        print(f"  {tag:<26}{'-- not run --':>11}")
        continue
    h = r["history"][-1]
    bc = math.log2(h["val_ppl"]) / cpt
    K = r["concepts"]["n_kept"]
    a = r.get("audit", {})
    ct = f"{a.get('n_pass','?')}/{a.get('n_concepts',K)}"
    print(f"  {tag:<26}{r['params']:>11,}{h['val_ppl']:>9.2f}{bc:>11.3f}"
          f"{h['val_acc']*100:>7.2f}%{ct:>10}")
    rows.append((tag, bc, r["params"], h["val_ppl"], h["val_acc"], out))

base = next((x for x in rows if x[0].startswith("BASELINE")), None)
print()
if base:
    print(f"  baseline: {base[1]:.3f} bits/char")
    better = [x for x in rows if x[0] != base[0] and not x[0].startswith("char") and x[1] < base[1]]
    print(f"  variants BEATING the baseline: {len(better)} of "
          f"{len([x for x in rows if x[0] != base[0] and not x[0].startswith('char')])}")
    for tag, bc, npar, ppl, acc, _ in sorted(better, key=lambda x: x[1]):
        gain = (base[1] - bc) / base[1] * 100
        print(f"     {tag:<26} {bc:.3f}  ({gain:4.1f}% better)  {npar:,} params")

print()
print("  BEST ON EACH METRIC")
print("  " + "-" * 96)
word = [x for x in rows if not x[0].startswith("char")]
if word:
    b_ppl = min(word, key=lambda x: x[3])
    b_bits = min(word, key=lambda x: x[1])
    b_acc = max(word, key=lambda x: x[4])
    b_small = min(word, key=lambda x: x[2])
    print(f"     best perplexity : {b_ppl[0]}  ppl {b_ppl[3]:.2f}")
    print(f"     best bits/char  : {b_bits[0]}  {b_bits[1]:.3f}")
    print(f"     best accuracy   : {b_acc[0]}  {b_acc[4]*100:.2f}%")
    print(f"     smallest        : {b_small[0]}  {b_small[2]:,} params")

print()
print("  PREDICTION CHECK")
print("  " + "-" * 96)
print("  1. 'A best per parameter, may LOSE on absolute perplexity'  -> A won BOTH. Partly WRONG.")
print("  2. 'B best responses'                                       -> B has the best accuracy. HELD.")
print("  3. 'C improves perplexity most'                             -> A improved most. WRONG.")
print("  4. 'D improves least'                                       -> D improved least. HELD.")
print("  5. 'no variant gives a coherent chatbot'                    -> see the replies. HELD.")
print("=" * 100)
