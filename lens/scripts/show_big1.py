"""show_big1.py — the wide-concept run, and whether the pre-registered predictions held.
"""
import json
import math
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat"
r = json.load(open(os.path.join(P, "big1_replies.json")))
r1 = json.load(open(os.path.join(P, "real1_replies.json")))
CPW = 60949068 / 13318447

h, h1 = r["history"][-1], r1["history"][-1]
a, a1 = r["audit"], r1["audit"]
K, K1 = r["concepts"]["n_kept"], r1["concepts"]["n_kept"]
ratio, ratio1 = 384 * 8 / K, 128 * 8 / K1
ct = f"{a['n_pass']}/{a['n_concepts']}"
ct1 = f"{a1['n_pass']}/{a1['n_concepts']}"
rate, rate1 = a["n_pass"] / a["n_concepts"], a1["n_pass"] / a1["n_concepts"]

print("=" * 96)
print("BIG1 vs RUN 1")
print("=" * 96)
print(f"  {'':<26}{'run 1':>16}{'big1':>16}")
print(f"  {'d':<26}{128:>16}{384:>16}")
print(f"  {'concepts kept':<26}{K1:>16}{K:>16}")
print(f"  {'bottleneck ratio':<26}{f'{ratio1:.0f}:1':>16}{f'{ratio:.0f}:1':>16}")
print(f"  {'parameters':<26}{r1['params']:>16,}{r['params']:>16,}")
print(f"  {'epochs':<26}{3:>16}{12:>16}")
print(f"  {'val perplexity':<26}{h1['val_ppl']:>16.2f}{h['val_ppl']:>16.2f}")
print(f"  {'val accuracy':<26}{f'{h1[chr(118)+chr(97)+chr(108)+chr(95)+chr(97)+chr(99)+chr(99)]*100:.2f}%':>16}"
      f"{f'{h[chr(118)+chr(97)+chr(108)+chr(95)+chr(97)+chr(99)+chr(99)]*100:.2f}%':>16}")
print(f"  {'bits per character':<26}{math.log2(h1['val_ppl'])/CPW:>16.3f}"
      f"{math.log2(h['val_ppl'])/CPW:>16.3f}")
print(f"  {'concepts passing':<26}{ct1:>16}{ct:>16}")
print(f"  {'pass RATE':<26}{f'{rate1*100:.1f}%':>16}{f'{rate*100:.1f}%':>16}")

print()
print("=" * 96)
print("PRE-REGISTERED PREDICTIONS — did they hold?")
print("=" * 96)
print(f"  1. 'perplexity falls well below 100'   ->  actual {h['val_ppl']:.2f}")
if h["val_ppl"] < 100:
    print("     HELD")
else:
    print(f"     WRONG. It fell 2.4x (353 -> {h['val_ppl']:.0f}) but did not reach 100.")
print("  2. 'replies become locally coherent, not topic-holding'  ->  see below")
print(f"  3. 'concept pass RATE rises with K'    ->  {rate1*100:.1f}% -> {rate*100:.1f}%")
if rate >= rate1:
    print("     HELD")
else:
    print("     WRONG. The COUNT rose (5 -> 16) but the RATE FELL.")

print()
print("  WHY THE RATE FELL, AND WHY THAT IS NOT THE WIDER LAYER FAILING:")
print("  " + "-" * 92)
fails = [c for c in a["concepts"] if not c["pass"]]
perf = [c for c in fails if c["auroc"] is not None and c["auroc"] > 0.99]
print(f"     failing concepts: {len(fails)} of {a['n_concepts']}")
print(f"     of those, AUROC > 0.99 (essentially perfect): {len(perf)}")
if fails:
    print(f"     their firing rates: {min(c['fires'] for c in fails):.3f} to "
          f"{max(c['fires'] for c in fails):.3f}   (floor 0.020)")
print()
print("     The added concepts are rare TOPICS -- family, money, travel, weather, technology,")
print("     animals, sport, music, art, law, politics, body, colour, education. They are learned")
print("     PERFECTLY (AUROC 1.0000) and still fail, because a concept firing on 0.5% of windows")
print("     is not doing work. THE FLOOR IS RIGHT AND DOES NOT MOVE.")
print()
print("     So: the wider layer raised the pass COUNT (5 -> 16) and raised perplexity and accuracy")
print("     substantially, but did NOT raise the pass RATE. Prediction 3 was WRONG as stated.")

print()
print("=" * 96)
print("WHAT IT SAYS NOW")
print("=" * 96)
for x in r["replies"][:4]:
    print(f"  PROMPT   {x['prompt'].splitlines()[0]}")
    print(f"  GREEDY   {x['greedy'][:150]}")
    print(f"  SAMPLED  {x['sampled'][:150]}")
    print()
print("=" * 96)
