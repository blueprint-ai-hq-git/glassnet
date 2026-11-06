"""get_vl_data.py — fetch image+caption pairs with an allowed licence, for the VL model.

THE VL IDEA
-----------
The image model already proved the fusion point: text is `16 tokens x D -> concept layer` and images
are `16 patches x D -> concept layer`. Same middle, two front-ends.

A VL model is therefore: ONE concept layer that has BOTH kinds of concept in it --
    image concepts (is_bright, left_right_symmetric, ...)
    text concepts  (is_a_question, mentions_a_colour, ...)
and is trained on real (image, caption) pairs. The image concepts describe what is IN the picture;
the text concepts describe what the caption SAYS. The shared layer is what relates them.

SOURCE (licence checked before download)
  HuggingFaceM4/COCO   CC BY 4.0   image/caption pairs
"""
import io
import json
import os
import urllib.request

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
OUT = os.path.join(PROJ, "data", "vl")
os.makedirs(OUT, exist_ok=True)


def api(repo, path=""):
    url = f"https://huggingface.co/api/datasets/{repo}" + (f"/tree/main/{path}" if path else "")
    req = urllib.request.Request(url, headers={"User-Agent": "GlassNet/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


print("=" * 100)
print("HuggingFaceM4/COCO — structure")
print("=" * 100)
try:
    top = api("HuggingFaceM4/COCO")
    print(f"  licence: {(top.get('cardData') or {}).get('license')}")
    names = [s["rfilename"] for s in top.get("siblings", [])]
    print(f"  files: {len(names)}")
    for n in names[:40]:
        print(f"    {n}")
    print()
    print("  --- subdirectories ---")
    for n in names:
        if n.endswith("/") or ("/" in n and n.count("/") == 1):
            print(f"    {n}")
except Exception as e:
    print(f"  ERROR {type(e).__name__}: {e}")

json.dump({"note": "structure probe"}, open(os.path.join(OUT, "probe.json"), "w"))
print("=" * 100)
