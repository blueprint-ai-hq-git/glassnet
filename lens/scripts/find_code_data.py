"""find_code_data.py — find CODE data with a license the project's rules allow.

The owner wants a model that can "talk code". Code is more learnable than general knowledge at small
scale: it is structured, deterministic, and has low-entropy patterns. So it is a fair target where
"knows everything" is not.

The rule stands: only MIT, Apache 2.0, CC0, or CC BY 4.0. Anything unclear is skipped and written
down as skipped. This checks the licence tag on each candidate BEFORE fetching anything.
"""
import json
import os
import urllib.request

ALLOWED = {"mit", "apache-2.0", "cc0-1.0", "cc-by-4.0", "cc-by-4", "apache 2.0", "mit license"}
CANDIDATES = [
    "codeparrot/codeparrot-clean",
    "bigcode/the-stack-smol",
    "bigcode/the-stack-dedup",
    "openai/openai_humaneval",
    "codeparrot/github-code-clean",
    "bigcode/starcoderdata",
    "nampdn-ai/tiny-codes",
    "ise-uiuc/Magicoder-OSS-Instruct-75K",
    "microsoft/CodeXGLUE",
    "google-deepmind/mbpp",
    "codeparrot/apps",
    "deepmind/code_contests",
    "open-r1/codeforces",
    "TIGER-Lab/CodeFeedback",
    "HuggingFaceH4/CodeAlpaca_20K",
]


def probe(repo):
    url = f"https://huggingface.co/api/datasets/{repo}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GlassNet/1.0"})
        with urllib.request.urlopen(req, timeout=45) as r:
            d = json.load(r)
    except Exception as e:
        return {"repo": repo, "error": f"{type(e).__name__}: {e}"}
    card = d.get("cardData") or {}
    lic = card.get("license")
    if isinstance(lic, list):
        lic = ",".join(str(x) for x in lic)
    files = [s["rfilename"] for s in d.get("siblings", [])]
    data = [f for f in files if f.endswith((".jsonl.gz", ".jsonl", ".parquet", ".json.gz", ".json"))]
    return {"repo": repo, "license": lic, "downloads": d.get("downloads"),
            "gated": d.get("gated"), "private": d.get("private"),
            "n_files": len(files), "data_files": data[:6]}


print("=" * 96)
print("CODE DATASETS — licence checked BEFORE anything is downloaded")
print("=" * 96)
print(f"  allowed licences: MIT, Apache 2.0, CC0, CC BY 4.0")
print()
usable = []
for repo in CANDIDATES:
    r = probe(repo)
    if "error" in r:
        print(f"  ????  {repo:<44} {r['error'][:40]}")
        continue
    lic = (r["license"] or "none-stated").lower()
    ok = lic in ALLOWED
    mark = "ALLOWED" if ok else "skip   "
    gated = " GATED" if r["gated"] else ""
    print(f"  {mark} {repo:<44} license={r['license']}{gated}")
    if ok and not r["gated"]:
        usable.append(r)

print()
print("=" * 96)
print(f"USABLE: {len(usable)}")
print("=" * 96)
for r in usable:
    print(f"  {r['repo']}  (downloads {r['downloads']})")
    for f in r["data_files"][:4]:
        print(f"      {f}")

with open("/run/media/yahya/HARD DISK 4/GlassNet/data/chat/code_sources.json", "w") as f:
    json.dump(usable, f, indent=1)
print()
print("  saved to data/chat/code_sources.json")
print("=" * 96)
