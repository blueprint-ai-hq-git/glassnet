"""t3_c_report.py — Target C: the readable-vs-normal verdict, on the frozen rule.

Readable seeds: seed 0 lives in the SCREEN file, seeds 1-2 in the CONFIRM file (the run's own
structure) -- both are read, or the mean would silently be a 2-seed mean.

WIN = readable beats the matched normal on BOTH perplexity AND accuracy by more than
2x the pooled std. Parity is never a win.
"""
import glob
import json
import os
import statistics as st

R = "runs/t3"


def load(f):
    return json.load(open(os.path.join(R, f)))


rd = [x for x in load("models_c_scr_lc0.1_lr1.0.json") if x["model"] == "readable"]
rd += [x for x in load("models_c_cfm_lc0.1_lr1.0.json") if x["model"] == "readable"]
nm = [x for x in load("models_c_normal.json") if x["model"] == "normal"]
rd.sort(key=lambda x: x["seed"])
nm.sort(key=lambda x: x["seed"])

print("=" * 96)
print("TARGET C — READABLE vs MATCHED NORMAL  (frozen rule: val ppl primary, acc secondary)")
print("=" * 96)
print(f"readable: {len(rd)} seeds, lam_concept=0.1, lr=1.0, params {rd[0]['params']}")
print(f"normal  : {len(nm)} seeds, lr=1.0,               params {nm[0]['params']}")
print(f"parameter ratio readable/normal = {rd[0]['params']/nm[0]['params']:.4f}")
print()
for tag, rows in (("readable", rd), ("normal", nm)):
    for x in rows:
        print(f"  {tag:<9} seed {x['seed']}  ppl {x['val_ppl']:8.4f}  acc {x['val']:6.2f}%  "
              f"best ep {x['best_epoch']:>3}  stopped {x['stopped_at']:>3}  {x['secs']:>5.0f}s")

rp = [x["val_ppl"] for x in rd]
np_ = [x["val_ppl"] for x in nm]
ra = [x["val"] for x in rd]
na = [x["val"] for x in nm]

print()
print(f"  readable mean ppl {st.mean(rp):8.4f}  sd {st.stdev(rp):6.4f}   "
      f"acc {st.mean(ra):6.2f}%  sd {st.stdev(ra):5.4f}")
print(f"  normal   mean ppl {st.mean(np_):8.4f}  sd {st.stdev(np_):6.4f}   "
      f"acc {st.mean(na):6.2f}%  sd {st.stdev(na):5.4f}")

ppl_pooled = ((st.stdev(rp) ** 2 + st.stdev(np_) ** 2) / 2) ** 0.5
acc_pooled = ((st.stdev(ra) ** 2 + st.stdev(na) ** 2) / 2) ** 0.5
d_ppl = st.mean(np_) - st.mean(rp)          # positive = readable BETTER (lower ppl)
d_acc = st.mean(ra) - st.mean(na)           # positive = readable BETTER

print()
print(f"  perplexity: readable is {'BETTER' if d_ppl > 0 else 'WORSE'} by {abs(d_ppl):.4f} "
      f"| pooled sd {ppl_pooled:.4f} | 2x pooled sd bar = {2*ppl_pooled:.4f} "
      f"-> {'BEATS' if d_ppl > 2*ppl_pooled else 'does NOT beat'} the bar")
print(f"  accuracy  : readable is {'BETTER' if d_acc > 0 else 'WORSE'} by {abs(d_acc):.4f} "
      f"| pooled sd {acc_pooled:.4f} | 2x pooled sd bar = {2*acc_pooled:.4f} "
      f"-> {'BEATS' if d_acc > 2*acc_pooled else 'does NOT beat'} the bar")

ppl_win, acc_win = d_ppl > 2 * ppl_pooled, d_acc > 2 * acc_pooled
if ppl_win and acc_win:
    verdict = "WIN"
elif abs(d_ppl) <= 2 * ppl_pooled and abs(d_acc) <= 2 * acc_pooled:
    verdict = "PARITY"
else:
    verdict = "BEHIND" if (d_ppl < 0 or d_acc < 0) else "MIXED (ahead on one metric only)"
print(f"\n  VERDICT: {verdict}   (parity is never a win)")

# ---- comparison against A3, the same task and the same budget -------------------------
a3 = load("a3_eval.json")
print("\n" + "=" * 96)
print("READABILITY PROGRESS — same task, same budget, same architecture")
print("=" * 96)
print(f"  A3 (word clusters)  readable ppl {a3.get('readable_ppl', 71.7980):8.4f}  "
      f"acc {a3.get('readable_acc', 30.39):6.2f}%")
print(f"  C  (real ideas)     readable ppl {st.mean(rp):8.4f}  acc {st.mean(ra):6.2f}%")
print(f"  change              ppl {st.mean(rp) - 71.7980:+.4f}   "
      f"acc {st.mean(ra) - 30.39:+.2f}")

json.dump({"readable": {"seeds": [x["seed"] for x in rd], "ppl": rp, "acc": ra,
                        "mean_ppl": st.mean(rp), "sd_ppl": st.stdev(rp),
                        "mean_acc": st.mean(ra), "sd_acc": st.stdev(ra),
                        "params": rd[0]["params"]},
           "normal": {"seeds": [x["seed"] for x in nm], "ppl": np_, "acc": na,
                      "mean_ppl": st.mean(np_), "sd_ppl": st.stdev(np_),
                      "mean_acc": st.mean(na), "sd_acc": st.stdev(na),
                      "params": nm[0]["params"]},
           "d_ppl": d_ppl, "d_acc": d_acc, "ppl_pooled": ppl_pooled, "acc_pooled": acc_pooled,
           "ppl_win": ppl_win, "acc_win": acc_win, "verdict": verdict},
          open(os.path.join(R, "c_eval.json"), "w"), indent=1)
print("\nsaved runs/t3/c_eval.json")
