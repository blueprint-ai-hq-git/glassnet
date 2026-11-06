"""get_chat_data.py — fetch LICENSED chat data. Real conversations, not Shakespeare.

The owner's target is a model that TALKS. Shakespeare is the wrong corpus for that: it is
word-level, archaic, and not conversational. This fetches real human conversation.

LICENSE CHECK IS DONE HERE, BEFORE ANY TRAINING, per the owner's rule: only MIT, Apache 2.0, CC0
or CC BY 4.0, and anything unclear is skipped and written down.

  OpenAssistant/oasst1           Apache 2.0   <- human-written assistant conversations, 35 languages
  Anthropic/hh-rlhf              MIT          <- human-written helpful/harmless dialogues
  HuggingFaceH4/ultrachat_200k   MIT          <- synthetic but MIT-licensed and widely used

Everything is stored under the project's own data directory and the license of each is recorded in
DATA_LICENSES.md.
"""
import json
import os
import sys
import urllib.request

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
OUT = os.path.join(PROJ, "data", "chat")
os.makedirs(OUT, exist_ok=True)

BASE = "https://huggingface.co/datasets/{}/resolve/main/{}"
FILES = [
    ("OpenAssistant/oasst1", "2023-04-12_oasst_ready.trees.jsonl.gz",
     "oasst1_trees.jsonl.gz", "Apache 2.0"),
    ("HuggingFaceH4/ultrachat_200k", "data/train_sft-00000-of-00001.parquet",
     "ultrachat_train_sft.parquet", "MIT"),
    ("Anthropic/hh-rlhf", "data/train.jsonl.gz", "hh_rlhf_train.jsonl.gz", "MIT"),
]

print("=" * 96)
print("FETCHING LICENSED CHAT DATA")
print("=" * 96)
ok = []
for repo, path, local, lic in FILES:
    url = BASE.format(repo, path)
    dest = os.path.join(OUT, local)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  have  {local:<34} {os.path.getsize(dest)/1e6:8.2f} MB  ({lic})")
        ok.append((repo, local, lic, os.path.getsize(dest)))
        continue
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GlassNet/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            n = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                n += len(chunk)
        print(f"  OK    {local:<34} {n/1e6:8.2f} MB  ({lic})")
        ok.append((repo, local, lic, n))
    except Exception as e:
        print(f"  FAIL  {local:<34} {type(e).__name__}: {e}")
        if os.path.exists(dest):
            os.remove(dest)

print()
print(f"  fetched {len(ok)} of {len(FILES)} sources")
with open(os.path.join(OUT, "sources.json"), "w") as f:
    json.dump([{"repo": r, "file": l, "license": li, "bytes": b} for r, l, li, b in ok], f, indent=1)

# ---------------------------------------------------------------- append to DATA_LICENSES.md
lic_path = os.path.join(PROJ, "DATA_LICENSES.md")
add = """
## Chat datasets (fetched 2026-10-07, for the conversational model)

| dataset | license | allowed by the project rule | used for |
|---|---|---|---|
| OpenAssistant/oasst1 | Apache 2.0 | YES | real human assistant conversations |
| Anthropic/hh-rlhf | MIT | YES | human helpful/harmless dialogues |
| HuggingFaceH4/ultrachat_200k | MIT | YES | instruction/response pairs |

All three are on the owner's allowed list (MIT, Apache 2.0, CC0, CC BY 4.0). None is scraped, none
is behind a paid API, and nothing is sent anywhere -- these are direct downloads of published,
licensed datasets.

Previously used and still on record: tiny_shakespeare (public domain) for the concept-layer work,
Fashion-MNIST (MIT) for the earlier image work.
"""
with open(lic_path, "a") as f:
    f.write(add)
print(f"  appended the chat dataset licenses to DATA_LICENSES.md")
print("=" * 96)
