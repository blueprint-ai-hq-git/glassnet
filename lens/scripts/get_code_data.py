"""get_code_data.py — fetch licensed code, and build a CHECKABLE code task.

WHY CODE IS A FAIR TARGET WHERE "KNOWS EVERYTHING" IS NOT
----------------------------------------------------------
Code is structured and deterministic. A small model can learn patterns like "a function that adds
two numbers" far more easily than it can learn world knowledge, because the mapping from problem to
answer is low-entropy and verifiable.

AND IT GIVES A CHECKABLE TEST. The owner's rule: "a CHECKABLE task, where a wrong answer is provably
wrong". For code that means: generate the function, EXECUTE it, compare the output to the expected
answer. No interpretation required -- it either runs and produces the right number or it does not.

SOURCES (licence checked before download, all on the allowed list)
  ise-uiuc/Magicoder-OSS-Instruct-75K   MIT        instruction -> code, JSONL
  codeparrot/apps                       MIT        problem -> solution, JSONL
"""
import json
import os
import urllib.request

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
OUT = os.path.join(PROJ, "data", "code")
os.makedirs(OUT, exist_ok=True)

BASE = "https://huggingface.co/datasets/{}/resolve/main/{}"
FILES = [
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", "data-oss_instruct-decontaminated.jsonl",
     "magicoder_oss_instruct.jsonl", "MIT"),
    ("codeparrot/apps", "train.jsonl", "apps_train.jsonl", "MIT"),
]

print("=" * 96)
print("FETCHING LICENSED CODE DATA")
print("=" * 96)
ok = []
for repo, path, local, lic in FILES:
    dest = os.path.join(OUT, local)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  have  {local:<34} {os.path.getsize(dest)/1e6:8.2f} MB  ({lic})")
        ok.append((repo, local, lic))
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
        ok.append((repo, local, lic))
    except Exception as e:
        print(f"  FAIL  {local:<34} {type(e).__name__}: {e}")

print()
print("  --- shape of the data ---")
mp = os.path.join(OUT, "magicoder_oss_instruct.jsonl")
if os.path.exists(mp):
    with open(mp) as f:
        for i, line in enumerate(f):
            if i >= 1:
                break
            d = json.loads(line)
            print(f"  magicoder keys: {sorted(d.keys())}")
            for k, v in d.items():
                if isinstance(v, str):
                    print(f"    {k}: {v[:220]!r}")
                elif isinstance(v, list):
                    print(f"    {k}: list of {len(v)}")
ap = os.path.join(OUT, "apps_train.jsonl")
if os.path.exists(ap):
    with open(ap) as f:
        d = json.loads(f.readline())
    print(f"  apps keys: {sorted(d.keys())}")

# ---------------------------------------------------------------- append the licence record
lic_path = os.path.join(PROJ, "DATA_LICENSES.md")
add = """
## Code datasets (fetched 2026-10-07, for the code half of the goal)

| dataset | license | allowed | used for |
|---|---|---|---|
| ise-uiuc/Magicoder-OSS-Instruct-75K | MIT | YES | instruction -> code pairs |
| codeparrot/apps | MIT | YES | problem -> solution, with test cases |
| codeparrot/github-code-clean | Apache 2.0 | YES (noted) | raw source files, 880 parquet shards |

Checked BEFORE download and recorded:
- **SKIPPED, licence unclear or absent:** codeparrot/codeparrot-clean (no licence stated),
  bigcode/the-stack-smol (no licence stated), bigcode/starcoderdata (licence "other"),
  HuggingFaceH4/CodeAlpaca_20K (licence "cc" -- not one of the allowed four).
- **SKIPPED, gated:** bigcode/the-stack-dedup, nampdn-ai/tiny-codes.
- **Unauthorized (401):** microsoft/CodeXGLUE, google-deepmind/mbpp, TIGER-Lab/CodeFeedback.
- **Allowed and available but not yet used:** openai/openai_humaneval (MIT),
  deepmind/code_contests (CC BY 4.0), open-r1/codeforces (CC BY 4.0).
"""
with open(lic_path, "a") as f:
    f.write(add)
print()
print("  licence record appended to DATA_LICENSES.md")
print("=" * 96)
