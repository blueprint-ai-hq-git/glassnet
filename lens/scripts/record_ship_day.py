"""record_ship_day.py — end-of-day record: K22 interface, second-dataset cleaner, caption cam."""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
cap = json.load(open(os.path.join(P, "lens", "runs", "video", "caption_cam_k22_log.json")))
cap12 = json.load(open(os.path.join(P, "lens", "runs", "video", "caption_cam2_log.json")))
cl = json.load(open(os.path.join(P, "lens", "runs", "clean", "apps_train_report.json")))

LOG = f"""
---

## 2026-10-08 late — THE SHIPPING STACK IS WIRED AND RE-DEMOED

### K=22 through the one interface
`variants.py` default see_talk is now `seetalk_k22_big.npz` (with fallback), and the smoke EVAL
loads whatever checkpoint the demo actually loaded (v._paths) — no hardcoded second path that
could score a different model than the one shown. Result:
```
  see_and_talk noun position, 500 TEST images: 401/500 = 80.2%  (K=12 small2 was 77.2%)
  explain now shows the extended concept names driving words: bottom_left_mass, centre_column...
```

### CAPTION CAMERA on the shipping model
```
  K=12 big:  {cap12['noun_correct']}/48 = {cap12['acc']*100:.1f}%   (caption_cam2.mp4)
  K=22 big:  {cap['noun_correct']}/48 = {cap['acc']*100:.1f}%   (caption_cam_k22.mp4)
```
Full-name greedy captions of real TEST frames, word by word, concepts on screen. +8.4 pts from the
shipping config, visible in the demo itself.

### THE CLEANER ON A SECOND DATASET (the "any dataset" claim, tested twice)
```
  magicoder_oss_instruct.jsonl (instruct schema):  75,197 -> 75,163 (99.95%), junk caught exactly
  apps_train.jsonl            (different schema):   5,000 ->  4,961 (99.22%)
      drop_repetition 27 · drop_low_signal 12 — sums to the 39 dropped, reconciled from the saved
      report JSON, thresholds frozen in the same file
```
load_any picked the `question` field automatically (its fallback chain: text/content/problem/
question/code/solution). One bug in my TEST script (wrong load_any signature — a returns-tuple
assumption) caught before results; clean.py itself untouched.

### LATEST TEST NUMBERS, ONE TABLE
```
  see-only (image cls)     read-auc'd concepts, 82-86% class band (image ladder)
  talk-only                chat model, 1.42 bits/char best arm (vocab 3.5k)
  see+talk K12             77.2% / 70.8% caption-cam nouns
  see+talk K22 (SHIP)      80.2% / 79.2% caption-cam nouns, delta +16.03 vs blind control
```
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended")
print(f"  interface 80.2% test | caption cam K22 {cap['acc']*100:.1f}% vs K12 "
      f"{cap12['acc']*100:.1f}% | apps dataset {cl['n_kept']}/{cl['n_in']} clean")
