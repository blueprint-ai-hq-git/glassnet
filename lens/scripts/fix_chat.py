"""fix_chat.py — see OASST1's real shape, and fetch the two datasets whose paths were wrong.
"""
import gzip
import json
import os
import urllib.request

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
CH = os.path.join(PROJ, "data", "chat")

print("=" * 96)
print("OASST1 — the real keys")
print("=" * 96)
p = os.path.join(CH, "oasst1_trees.jsonl.gz")
with gzip.open(p, "rt") as f:
    first = f.readline()
row = json.loads(first)
print(f"  top-level keys: {sorted(row.keys())}")
print(f"\n  --- raw first row, truncated ---")
print(json.dumps(row, indent=1)[:1400])

print()
print("=" * 96)
print("FETCHING THE TWO THAT 404'd — correct paths from the HF API")
print("=" * 96)
BASE = "https://huggingface.co/datasets/{}/resolve/main/{}"
FILES = [
    ("Anthropic/hh-rlhf", "helpful-base/train.jsonl.gz", "hh_rlhf_helpful_train.jsonl.gz", "MIT"),
    ("Anthropic/hh-rlhf", "harmless-base/train.jsonl.gz", "hh_rlhf_harmless_train.jsonl.gz", "MIT"),
]
for repo, path, local, lic in FILES:
    dest = os.path.join(CH, local)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  have  {local:<34} {os.path.getsize(dest)/1e6:8.2f} MB  ({lic})")
        continue
    try:
        req = urllib.request.Request(BASE.format(repo, path), headers={"User-Agent": "GlassNet/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
            n = 0
            while True:
                c = r.read(1 << 20)
                if not c:
                    break
                f.write(c)
                n += len(c)
        print(f"  OK    {local:<34} {n/1e6:8.2f} MB  ({lic})")
    except Exception as e:
        print(f"  FAIL  {local:<34} {type(e).__name__}: {e}")

print()
print("  --- hh-rlhf shape ---")
hp = os.path.join(CH, "hh_rlhf_helpful_train.jsonl.gz")
if os.path.exists(hp):
    with gzip.open(hp, "rt") as f:
        r0 = json.loads(f.readline())
    print(f"  keys: {sorted(r0.keys())}")
    print(f"  chosen[:300]: {r0['chosen'][:300]!r}")
print("=" * 96)
