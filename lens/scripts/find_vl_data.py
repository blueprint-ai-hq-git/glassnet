"""find_vl_data.py — find IMAGE+CAPTION data that is fetchable and licensed.

HuggingFaceM4/COCO turned out to be a loading script (COCO.py) with the images hosted elsewhere, so
it is not directly usable. This probes a wider list and reports which ones have real downloadable
files AND an allowed licence.
"""
import json
import os
import urllib.request

ALLOWED = {"mit", "apache-2.0", "cc0-1.0", "cc-by-4.0", "cc-by-4", "apache 2.0", "mit license",
           "cc0 1.0", "cc by 4.0", "cc-by-sa-4.0"}

CANDIDATES = [
    "textvqa",
    "lmms-lab/textvqa",
    "facebook/textvqa",
    "nlphuji/flickr30k",
    "HuggingFaceM4/VQAv2",
    "lmms-lab/VQAv2",
    "lmms-lab/DocVQA",
    "merve/vqav2-small",
    "nateraw/pascal-voc-2012",
    "clip-benchmark/wds_flickr30k",
    "clip-benchmark/wds_coco_captioning",
    "nlphuji/coco_karpathy",
    "sayakpaul/coco-30-val-2014",
    "microsoft/azure-coco",
    "conceptual_captions",
    "google-research-datasets/conceptual_captions",
    "MMMU/MMMU",
    "lmms-lab/MMMU",
    "openbmb/VisIT-Bench",
    "lmms-lab/VizWiz",
    "HuggingFaceM4/OBELICS",
    "Salesforce/blip3-ocr-200m",
    "deepmind/code_contests",
]


def probe(repo):
    url = f"https://huggingface.co/api/datasets/{repo}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GlassNet/1.0"})
        with urllib.request.urlopen(req, timeout=40) as r:
            d = json.load(r)
    except Exception as e:
        return {"repo": repo, "error": type(e).__name__}
    card = d.get("cardData") or {}
    lic = card.get("license")
    if isinstance(lic, list):
        lic = ",".join(str(x) for x in lic)
    files = [s["rfilename"] for s in d.get("siblings", [])]
    real = [f for f in files if f.endswith((".parquet", ".jsonl", ".json", ".csv", ".tsv",
                                            ".jsonl.gz", ".arrow", ".zip", ".tar", ".npz"))]
    return {"repo": repo, "license": lic, "gated": d.get("gated"), "n_files": len(files),
            "data_files": real[:5], "has_script": any(f.endswith(".py") for f in files)}


print("=" * 100)
print("VL DATASETS — fetchable files + allowed licence")
print("=" * 100)
usable = []
for repo in CANDIDATES:
    r = probe(repo)
    if "error" in r:
        print(f"  ????  {repo:<42} {r['error']}")
        continue
    lic = (r["license"] or "none-stated").lower()
    ok = lic in ALLOWED
    ndata = len(r["data_files"])
    mark = "ALLOWED" if ok else "skip   "
    g = " GATED" if r["gated"] else ""
    sc = " script-only" if (r["has_script"] and ndata == 0) else ""
    print(f"  {mark} {repo:<42} license={r['license']}{g}{sc}")
    if ok and not r["gated"] and ndata > 0:
        usable.append(r)

print()
print("=" * 100)
print(f"USABLE AND FETCHABLE: {len(usable)}")
print("=" * 100)
for r in usable:
    print(f"  {r['repo']}  ({r['n_files']} files)")
    for f in r["data_files"][:4]:
        print(f"      {f}")

path = "/run/media/yahya/HARD DISK 4/GlassNet/data/vl_sources.json"
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(usable, open(path, "w"), indent=1)
print()
print("  saved to data/vl_sources.json")
print("=" * 100)
