#!/usr/bin/env python3
"""opt_vs_sgd_table.py -- the comparison TABLE for PLAN item (d2b).

Reads the arms' OWN append-only per-epoch rows (lens/runs/chat/opt_<arm>_s<seed>_history.jsonl, written
by train_chat.py every epoch) and prints, for every arm and seed: epochs completed, the assistant-masked
val accuracy at epoch 12 and at its best epoch, the masked val perplexity, the step size actually used,
seconds per epoch, and whether every printed number is finite.

It then prints the numbers the PREDICTIONS in lens/runs/OPT-VS-SGD-PREDICTIONS.txt compare, side by side
with the bar, so no claim has to be checked by eye:
  P42d  best relgrad arm vs best SGD arm (within 1.0 point?)
  P42e  spread across the three relgrad step sizes vs the spread across the two SGD step sizes (<= 3.0?)
  P42f  sgd@1.0 vs sgd@0.4 (at least 2 points below, or non-finite?)
  P42g  relgrad@0.05 constant step vs with decay, seed 0
  P42b  finiteness of every relgrad row
It prints the COMPARED VALUES beside each verdict. It does not decide anything the predictions did not
already fix, and it makes no accuracy claim beyond the clean held-out split it names.
"""
import glob
import json
import os
import re

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
CHAT = os.path.join(PROJ, "lens", "runs", "chat")
ARMS = ["sgd04", "sgd10", "relgrad005", "relgrad05", "relgrad5", "relgrad05c"]
REL_LRS = {"relgrad005": 0.005, "relgrad05": 0.05, "relgrad5": 0.5}


def load():
    runs = {}
    for path in sorted(glob.glob(os.path.join(CHAT, "opt_*_s*_history.jsonl"))):
        m = re.match(r"opt_(.+)_s(\d+)_history\.jsonl$", os.path.basename(path))
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        if arm not in ARMS:
            continue
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        # A RE-RUN OF THE SAME ARM APPENDS (train_chat.py opens this log with "a"), so a truncated arm that
        # was re-queued by hand leaves its stale rows IN FRONT of the fresh ones (measured 2026-10-21:
        # sgd04 seed 1 was killed after epoch 8 by an OOM and re-queued). Keep the LAST row per epoch (the
        # newest run wins) and sort by epoch, so "epochs completed" is the number of DISTINCT epochs and
        # never a row count that double-counts a re-run. For a file with no duplicates this is the identity,
        # so no stored table number moves.
        by_ep = {}
        for r in rows:
            by_ep[int(r["epoch"])] = r
        rows = [by_ep[k] for k in sorted(by_ep)]
        runs[(arm, seed)] = (rows, path)
    return runs


def fin(x):
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def masked(r):
    return r.get("val_masked_acc")


def main():
    runs = load()
    if not runs:
        print("NO ARM ROWS FOUND under lens/runs/chat/opt_*_s*_history.jsonl -- nothing has run yet.")
        return
    print(f"arms found: {len(runs)}   source: lens/runs/chat/opt_<arm>_s<seed>_history.jsonl")
    print("metric: assistant-masked val accuracy (train_chat.py 'val_masked_acc'), val split which=1 of")
    print("data/chat/prepared_mix_ctrl_honest (VERDICT CLEAN: disjoint from train at the training window)")
    print()
    hdr = (f"{'arm':<12}{'seed':>5}{'ep_done':>8}{'masked@ep12':>13}{'best masked':>13}{'best ep':>8}"
           f"{'masked ppl@ep12':>17}{'lr@ep12':>12}{'sec/ep':>9}{'all finite':>11}")
    print(hdr)
    print("-" * len(hdr))
    for arm in ARMS:
        for seed in sorted(s for (a, s) in runs if a == arm):
            rows, _ = runs[(arm, seed)]
            ep12 = next((r for r in rows if r["epoch"] == 12), None)
            best = max(rows, key=lambda r: (masked(r) if masked(r) is not None else -1))
            finite = all(fin(r.get("train_loss")) and fin(r.get("val_ppl")) and fin(masked(r))
                         for r in rows)
            secs = [r.get("seconds", 0) for r in rows]
            m12 = f"{masked(ep12)*100:.2f}%" if ep12 is not None and masked(ep12) is not None else "--"
            mbest = f"{masked(best)*100:.2f}%" if masked(best) is not None else "--"
            ppl12 = f"{ep12.get('val_masked_ppl'):,.1f}" if ep12 is not None else "--"
            lr12 = f"{ep12.get('lr'):.6f}" if ep12 is not None else "--"
            secs_txt = f"{sum(secs)/max(len(secs),1):.0f}" if secs else "--"
            print(f"{arm:<12}{seed:>5}{len(rows):>8}{m12:>13}{mbest:>13}{best['epoch']:>8}"
                  f"{ppl12:>17}{lr12:>12}{secs_txt:>9}{('yes' if finite else 'NO'):>11}")
    print()
    print("=== THE BARS FROM lens/runs/OPT-VS-SGD-PREDICTIONS.txt, WITH THE COMPARED VALUES ===")

    def ep12(arm, seed):
        rows = runs.get((arm, seed), (None, None))[0]
        if not rows:
            return None
        r = next((x for x in rows if x["epoch"] == 12), None)
        return masked(r) if r else None

    def best_any(arm):
        vals = [ep12(arm, s) for (a, s) in runs if a == arm]
        vals = [v for v in vals if v is not None]
        return (max(vals) if vals else None), len(vals)

    rel_best = max(((v, a) for a in REL_LRS for v, n in [best_any(a)] if v is not None), default=None)
    sgd_best = max(((v, a) for a in ("sgd04", "sgd10") for v, n in [best_any(a)] if v is not None),
                   default=None)
    if rel_best and sgd_best:
        gap = (sgd_best[0] - rel_best[0]) * 100
        print(f"P42d best relgrad {rel_best[1]} {rel_best[0]*100:.2f}%  vs  best SGD {sgd_best[1]} "
              f"{sgd_best[0]*100:.2f}%   -> relgrad is {gap:.2f} points BELOW SGD; bar: within 1.00 "
              f"-> {'PASS' if gap <= 1.0 else 'FAIL'}")
    else:
        print("P42d not decidable yet (missing arms)")

    rel_vals = {a: ep12(a, 0) for a in REL_LRS}
    rel_vals = {a: v for a, v in rel_vals.items() if v is not None}
    sgd_vals = {a: ep12(a, 0) for a in ("sgd04", "sgd10")}
    sgd_vals = {a: v for a, v in sgd_vals.items() if v is not None}
    if len(rel_vals) >= 2:
        rel_spread = (max(rel_vals.values()) - min(rel_vals.values())) * 100
        print(f"P42e seed 0 relgrad ep12: " + ", ".join(f"{a} {v*100:.2f}%" for a, v in rel_vals.items())
              + f"  -> spread {rel_spread:.2f} points")
    if len(sgd_vals) == 2:
        sgd_spread = (max(sgd_vals.values()) - min(sgd_vals.values())) * 100
        print(f"P42e seed 0 SGD ep12:     " + ", ".join(f"{a} {v*100:.2f}%" for a, v in sgd_vals.items())
              + f"  -> spread {sgd_spread:.2f} points")
    if len(rel_vals) >= 2 and len(sgd_vals) == 2:
        rel_spread = (max(rel_vals.values()) - min(rel_vals.values())) * 100
        sgd_spread = (max(sgd_vals.values()) - min(sgd_vals.values())) * 100
        print(f"P42e bar: relgrad spread <= 3.00 AND strictly smaller than the SGD spread "
              f"({rel_spread:.2f} vs {sgd_spread:.2f}) -> "
              f"{'PASS' if rel_spread <= 3.0 and rel_spread < sgd_spread else 'FAIL'}")
    if "sgd04" in sgd_vals and "sgd10" in sgd_vals:
        d = (sgd_vals["sgd04"] - sgd_vals["sgd10"]) * 100
        print(f"P42f sgd04 {sgd_vals['sgd04']*100:.2f}% vs sgd10 {sgd_vals['sgd10']*100:.2f}% -> "
              f"difference {d:.2f} points; bar: sgd10 at least 2.00 below (or non-finite) -> "
              f"{'PASS' if d >= 2.0 else 'FAIL'}")
    d_dec, d_con = ep12("relgrad05", 0), ep12("relgrad05c", 0)
    if d_dec is not None and d_con is not None:
        print(f"P42g relgrad05 decay 0.02 {d_dec*100:.2f}% vs constant step {d_con*100:.2f}% (seed 0) -> "
              f"decay is {(d_dec-d_con)*100:+.2f} points; bar: decay strictly better -> "
              f"{'PASS' if d_dec > d_con else 'FAIL'}")
    rel_rows = {k: v for k, v in runs.items() if k[0].startswith("relgrad")}
    rel_bad = [(a, s) for (a, s), (rows, _) in rel_rows.items()
               if not all(fin(r.get("train_loss")) and fin(masked(r)) for r in rows)]
    if not rel_rows:
        print("P42b NO relgrad arm has run yet -> NOT DECIDABLE (not a PASS: nothing has been measured)")
    else:
        print(f"P42b relgrad arms measured: {len(rel_rows)}; arms with a non-finite number: "
              f"{rel_bad if rel_bad else 'none'} -> {'PASS' if not rel_bad else 'FAIL'}")
    print()
    print("NOT CLAIMED: nothing here is a chat-quality claim. The metric is next-token accuracy on the")
    print("assistant's own turn, held-out and clean; the memorisation gate is OFF in every arm (P42i).")


if __name__ == "__main__":
    main()
