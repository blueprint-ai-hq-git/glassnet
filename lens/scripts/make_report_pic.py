#!/usr/bin/env python3
"""make_report_pic.py — one figure with the two ladders that actually decided this week.
Every number is labelled with the file it came from; nothing is estimated for the picture.
Sources: lens/runs/BENCH.md (honest ladder + cross-corpus), lens/runs/HONEST-LADDER.txt,
         /tmp/talk_quality_controls.log (talk ladder), lens/runs/BENCH_EXTERNAL.json (external bench)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/HONEST-PICTURE.png"
plt.rcParams.update({"font.size": 10})

SHIPS = [
    ("t11 d256\n192 ep", 54.89, 7.62),
    ("t12 d512\n192 ep", 56.07, 8.18),
    ("t12 splitfix\nd512 192 ep", 63.85, 8.32),
    ("OURS S0 d256\n96 ep", None, 9.84),
    ("OURS B0 d512\n96 ep", None, 8.86),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))

# --- panel A: what we thought vs what is true ------------------------------------------------
x = range(len(SHIPS))
rec = [s[1] or 0 for s in SHIPS]
hon = [s[2] for s in SHIPS]
ax1.bar([i - 0.2 for i in x], rec, width=0.4, label="as recorded (memorised val windows included)",
        color="#c44e52")
ax1.bar([i + 0.2 for i in x], hon, width=0.4, label="honest (val windows seen in training removed)",
        color="#4c72b0")
for i, s in enumerate(SHIPS):
    if s[1]:
        ax1.text(i - 0.2, s[1] + 1.2, f"{s[1]:.2f}%", ha="center", fontsize=9)
    else:
        ax1.text(i - 0.2, 1.2, "never\nrecorded", ha="center", fontsize=8, color="#555")
    ax1.text(i + 0.2, s[2] + 1.2, f"{s[2]:.2f}%", ha="center", fontsize=9)
ax1.set_xticks(list(x))
ax1.set_xticklabels([s[0] for s in SHIPS], fontsize=9)
ax1.set_ylabel("assistant-masked next-token accuracy (%)")
ax1.set_title("Talk accuracy: every model is 7.6-10% once the val is really held out\n"
              "source: lens/runs/BENCH.md + HONEST-LADDER.txt (n=709 old ships, n=11,519 ours)",
              fontsize=10)
ax1.legend(fontsize=8, loc="upper left")
ax1.set_ylim(0, 72)

# --- panel B: talk quality meter with its own controls ---------------------------------------
labels = ["real held-out\ntext", "same text,\nwords shuffled", "BAR = midpoint", "1-epoch\nmodel",
          "8-epoch\nmodel", "shipped\nmodel"]
vals = [-3.63, -12.24, -7.94, -12.83, -8.88, -11.50]
cols = ["#55a868", "#c44e52", "#333333", "#c44e52", "#c44e52", "#c44e52"]
ax2.barh(labels, vals, color=cols)
for i, v in enumerate(vals):
    ax2.text(v - 0.35, i, f"{v:.2f}", va="center", ha="right", fontsize=9)
ax2.axvline(-7.94, color="#333333", linestyle="--", linewidth=1.4)
ax2.set_xlabel("trigram log-probability per token (higher = more like real text)")
ax2.set_title("Text quality: every model we have is word salad\n"
              "source: /tmp/talk_quality_controls.log (real/shuffled/bar controls run 2026-10-16)",
              fontsize=10)
ax2.set_xlim(-14.5, -2.5)

fig.suptitle("GlassNet honest picture, 2026-10-17 10:35 IST - all numbers measured, none estimated",
             fontsize=12, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=115)
print("wrote", OUT)

# external bench, printed for the same picture (numbers live in BENCH_EXTERNAL.json)
b = json.load(open("/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/BENCH_EXTERNAL.json"))
print(f"external: MMLU {b['mmlu']['acc']*100:.1f}% (n={b['mmlu']['n']}, chance "
      f"{b['mmlu']['random_baseline']*100:.0f}%, mean prob of the correct choice "
      f"{b['mmlu']['mean_prob_of_correct_choice']:.5f}) | GSM8K {b['gsm8k']['acc']*100:.1f}% "
      f"(n={b['gsm8k']['n']}) | code syntax-valid {b['code']['syntax_valid_rate']*100:.1f}% "
      f"(n={b['code']['n']})")