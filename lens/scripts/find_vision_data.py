"""find_vision_data.py — find FACE and IMAGE-CAPTION data with an allowed licence.

The owner's rule is unchanged: MIT, Apache 2.0, CC0, CC BY 4.0 only. Anything unclear, gated,
non-commercial, or absent is SKIPPED and written down with the reason.

Three groups needed:
  FACES      -- for the face detection / recognition test
  CAPTIONS   -- image -> text, for the captioning model
  IMAGES     -- general image concepts, where a safe licence already exists

Checked BEFORE anything is downloaded.
"""
import json
import os
import urllib.request

ALLOWED = {"mit", "apache-2.0", "cc0-1.0", "cc-by-4.0", "cc-by-4", "apache 2.0", "mit license",
           "cc0 1.0", "cc by 4.0", "apache-2.0 license", "mit-0"}

GROUPS = {
    "FACES": [
        "nvidia/digiface-1m",
        "microsoft/digiface",
        "Dagnostic/UTKFace",
        "visforai/UTKFace",
        "nateraw/lfw",
        "logasja/lfw",
        "ylecun/mnist",
        "nielsr/ffhq",
        "pokej/ffhq",
        "huggan/ffhq",
    ],
    "CAPTIONS": [
        "nlphuji/flickr30k",
        "HuggingFaceM4/COCO",
        "lmms-lab/COCO-Caption",
        "conceptual_captions",
        "sayakpaul/coco-30-val-2014",
        "HuggingFaceM4/VQAv2",
        "textvqa",
        "nlpconnect/vit-gpt2-image-captioning",
        "microsoft/common_voice",
    ],
    "IMAGES_SAFE": [
        "medmnist/medmnist",
        "albertvillanova/medmnist-v2",
        "zalando-datasets/fashion_mnist",
        "uoft-cs/cifar10",
        "clane9/imagenet-100",
    ],
}


def probe(repo):
    url = f"https://huggingface.co/api/datasets/{repo}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GlassNet/1.0"})
        with urllib.request.urlopen(req, timeout=40) as r:
            d = json.load(r)
    except Exception as e:
        return {"repo": repo, "error": f"{type(e).__name__}"}
    card = d.get("cardData") or {}
    lic = card.get("license")
    if isinstance(lic, list):
        lic = ",".join(str(x) for x in lic)
    files = [s["rfilename"] for s in d.get("siblings", [])]
    data = [f for f in files if f.endswith((".parquet", ".jsonl", ".csv", ".npz", ".tar", ".zip",
                                            ".json", ".tsv", ".arrow"))]
    return {"repo": repo, "license": lic, "gated": d.get("gated"), "downloads": d.get("downloads"),
            "n_files": len(files), "data_files": data[:5]}


print("=" * 100)
print("FINDING VISION DATA — licence checked BEFORE download")
print("=" * 100)
print("  allowed: MIT, Apache 2.0, CC0, CC BY 4.0")
print()
out = {}
for group, repos in GROUPS.items():
    print(f"--- {group} " + "-" * (94 - len(group)))
    usable = []
    for repo in repos:
        r = probe(repo)
        if "error" in r:
            print(f"    ????  {repo:<38} {r['error']}")
            continue
        lic = (r["license"] or "none-stated").lower()
        ok = lic in ALLOWED
        mark = "ALLOWED" if ok else "skip   "
        g = " GATED" if r["gated"] else ""
        print(f"    {mark} {repo:<38} license={r['license']}{g}")
        if ok and not r["gated"]:
            usable.append(r)
    out[group] = usable
    print()

print("=" * 100)
print("USABLE")
print("=" * 100)
for group, us in out.items():
    print(f"  {group}: {len(us)}")
    for r in us:
        print(f"     {r['repo']}  (downloads {r['downloads']})")
        for f in r["data_files"][:3]:
            print(f"         {f}")

path = "/run/media/yahya/HARD DISK 4/GlassNet/data/vision_sources.json"
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(out, open(path, "w"), indent=1)
print()
print(f"  saved to data/vision_sources.json")
print("=" * 100)
