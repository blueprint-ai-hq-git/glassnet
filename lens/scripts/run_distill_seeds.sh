#!/bin/bash
# run_distill_seeds.sh — the distillation result needs 3 seeds before it is called a win.
#
# Seed 0: student-A 105.357 / 29.78%   student-B 107.353 / 29.65%   teacher 107.535 / 29.53%
#   -> A beats B on both metrics, and A beats its own teacher at 1/4 the size.
#   BUT the accuracy margin is +0.125%, which is inside seed noise, and it is ONE seed.
#
# So: two more seeds. If the direction holds on all three, it is a win. If it flips on one, the
# honest verdict is NO DIFFERENCE, and that is what gets recorded.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export DS_DEVICE=gpu
export DS_EPOCHS="${DS2_EPOCHS:-40}"

for s in 1 2; do
  echo
  echo "################################################################"
  echo "# DISTILLATION SEED $s"
  echo "################################################################"
  DS_SEED=$s DS_OUT="distill_seed$s" nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/run_distill.py 2>&1 | tail -14
done

echo
echo "################################################################"
echo "# DISTILLATION — 3 SEEDS"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os, statistics as st
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs"
rows = []
for tag, d in [("seed 0", "distill"), ("seed 1", "distill_seed1"), ("seed 2", "distill_seed2")]:
    f = f"{P}/{d}/verdict.json"
    if not os.path.exists(f):
        print(f"  {tag}: -- not run --")
        continue
    v = json.load(open(f))
    if v.get("verdict") == "NO VERDICT":
        print(f"  {tag}: NO VERDICT (teacher below bar)")
        continue
    rows.append((tag, v))
    print(f"  {tag}  teacher {v['teacher']['acc']*100:6.2f}%  "
          f"A {v['student_a']['acc']*100:6.2f}% / ppl {v['student_a']['ppl']:7.3f}   "
          f"B {v['student_b']['acc']*100:6.2f}% / ppl {v['student_b']['ppl']:7.3f}   "
          f"d_acc {v['delta_acc']*100:+.3f}%  d_ppl {v['delta_ppl']:+.3f}")

if len(rows) >= 2:
    da = [v["delta_acc"] for _, v in rows]
    dp = [v["delta_ppl"] for _, v in rows]
    print()
    print(f"  A-vs-B accuracy delta: mean {st.mean(da)*100:+.3f}%  std {st.pstdev(da)*100:.3f}%")
    print(f"  A-vs-B perplexity delta: mean {st.mean(dp):+.3f}  std {st.pstdev(dp):.3f}")
    wins_a = sum(1 for _, v in rows if v["delta_acc"] > 0)
    wins_p = sum(1 for _, v in rows if v["delta_ppl"] < 0)
    print(f"  seeds where A beats B on accuracy:  {wins_a}/{len(rows)}")
    print(f"  seeds where A beats B on perplexity: {wins_p}/{len(rows)}")
    if wins_a == len(rows) and wins_p == len(rows):
        print("\n  VERDICT: THE DISTILLED STUDENT WINS ON EVERY SEED, ON BOTH METRICS.")
    elif wins_p == len(rows):
        print("\n  VERDICT: the distilled student wins on PERPLEXITY every seed, but the ACCURACY")
        print("  direction is not unanimous. Reported as: distillation helps perplexity, accuracy")
        print("  inconclusive at this size.")
    else:
        print("\n  VERDICT: NO CONSISTENT DIFFERENCE. The single-seed result did not survive seeds.")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo "docs copied"
