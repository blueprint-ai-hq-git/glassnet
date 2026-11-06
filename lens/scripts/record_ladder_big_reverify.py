"""record_ladder_big_reverify.py — the corrected big-rung numbers (readable arm, by label).
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
new = json.load(open(os.path.join(P, "lens", "runs", "ladder", "ladder_big.json")))
old = json.load(open(os.path.join(P, "lens", "runs", "ladder",
                                  "ladder_big_prior_plainparsed.json")))


def row(d, cap):
    for r in d:
        if r["capability"] == cap:
            return r
    return {}


LOG = f"""
---

## 2026-10-08 19:57 — BIG RUNG RE-VERIFIED after the wrong-subject fix: 7/7, image_cls 83.49%

Full big rung re-run (24 epochs, 60k, GPU, all seven capabilities) with `readable_test()` parsing
BY LABEL. Prior artifact preserved as `ladder_big_prior_plainparsed.json` — nothing overwritten.

```
  capability   prior (plain-parsed, WRONG SUBJECT)   re-verified (readable arm, BY LABEL)
  image_cls    86.42%                                83.49%   <- the honest number
  text_lm      ppl 142.35 -> PASS                    ppl 142.35   PASS
  vl_vqa       83.36%                               83.36%   PASS
  code_lm      ppl 50.2                              ppl 50.2     PASS
  gradcheck / explain / audit                        PASS (0.003457 / 4.77e-07 / 3 of 12)
  RUNG 'big': ALL PASS 7/7
```

**Read this correctly: the fix moved the image_cls number DOWN by 2.93 points, because the old
86.42% was the plain control's score being reported as the readable model's.** 83.49% is what the
readable K=12 model actually achieves at big scale. Every other arm is unchanged (they were parsed
correctly — their gates read labelled lines or their own final printout), so the regression is
clean: nothing else broke, and the one number that was wrong is now right and lower.

The corrected ladder therefore still shows the monotone climb (tiny 44.8 -> small 65.76 -> medium
82.52 -> big 83.49 readable) and the six other capabilities are unaffected.

TIMING (real, from this run): gradcheck 0.3s · explain 1.2s · audit 0.6s · text_lm 486s ·
image_cls 41s · vl_vqa 48s · code_lm 361s — full big rung ~16 min wall on the GTX 1660S.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended")
print("  image_cls  prior(plain) 86.42%  ->  readable 83.49%   (fix moved it DOWN: honest)")
for c in ("text_lm", "vl_vqa", "code_lm"):
    print(f"  {c:<9} {row(new, c)['checks']}")
