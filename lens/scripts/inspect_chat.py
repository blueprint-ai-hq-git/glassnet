"""inspect_chat.py — what is actually in the chat data, and the right paths for the others.
"""
import gzip
import json
import os
import urllib.request

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
CH = os.path.join(PROJ, "data", "chat")

print("=" * 96)
print("OASST1 — what is in it")
print("=" * 96)
p = os.path.join(CH, "oasst1_trees.jsonl.gz")
rows = []
with gzip.open(p, "rt") as f:
    for i, line in enumerate(f):
        if i >= 200000:
            break
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
print(f"  trees loaded: {len(rows):,}")

# OASST1 is a tree per conversation; walk a couple to see the shape
def walk(node, depth=0, out=None):
    if out is None:
        out = []
    out.append((depth, node.get("role"), (node.get("text") or "")[:70]))
    for ch in node.get("replies", [])[:1]:
        walk(ch, depth + 1, out)
    return out


print("\n  --- one conversation tree ---")
for r in rows[:2]:
    for depth, role, text in walk(r):
        print(f"    {'  '*depth}[{role}] {text!r}")

roles = {}
langs = {}
for r in rows:
    def rec(n):
        roles[n.get("role")] = roles.get(n.get("role"), 0) + 1
        langs[n.get("lang")] = langs.get(n.get("lang"), 0) + 1
        for ch in n.get("replies", []):
            rec(ch)
    rec(r)
print(f"\n  roles: {roles}")
top = sorted(langs.items(), key=lambda x: -x[1])[:6]
print(f"  top languages: {top}")

# how many assistant replies exist in English
en_asst = 0
for r in rows:
    def rec(n):
        global en_asst
        if n.get("role") == "assistant" and n.get("lang") == "en" and (n.get("text") or "").strip():
            en_asst += 1
        for ch in n.get("replies", []):
            rec(ch)
    rec(r)
print(f"  English assistant messages: {en_asst:,}")

print()
print("=" * 96)
print("THE OTHER TWO — finding the real file paths via the HF API")
print("=" * 96)
for repo in ["HuggingFaceH4/ultrachat_200k", "Anthropic/hh-rlhf"]:
    try:
        url = f"https://huggingface.co/api/datasets/{repo}"
        req = urllib.request.Request(url, headers={"User-Agent": "GlassNet/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        files = [s["rfilename"] for s in d.get("siblings", [])]
        data = [f for f in files if f.endswith((".jsonl.gz", ".parquet", ".jsonl", ".json.gz"))]
        print(f"\n  {repo}  (license tag: {d.get('cardData', {}).get('license', 'see README')})")
        for f in data[:8]:
            print(f"      {f}")
    except Exception as e:
        print(f"\n  {repo}: {type(e).__name__}: {e}")
print("=" * 96)
