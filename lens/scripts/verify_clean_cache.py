"""verify_clean_cache.py — confirm code_data.build() finds the CLEAN corpus cache, so training does
not silently rebuild a different corpus.
"""
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
import code_data                                          # noqa: E402

d = code_data.build()
v, ids = d["vocab"], d["ids"]
bt = sum(1 for i in ids if v[i] == "`")
print(f"  DOC_MODE       : {code_data.DOC_MODE}")
print(f"  vocab          : {len(v):,}")
print(f"  docs           : {len(d['bounds']) - 1:,}")
print(f"  tokens         : {len(ids):,}")
print(f"  backticks      : {bt:,} ({bt/len(ids)*100:.4f}%)")
print(f"  first doc      : {' '.join(v[i] for i in ids[d['bounds'][0]:d['bounds'][0] + 40])[:150]}")
ok = code_data.DOC_MODE == "clean" and bt / len(ids) < 0.002
print(f"\n  {'PASS: clean corpus loaded, cache hit' if ok else 'FAIL: wrong corpus — do not train'}")
sys.exit(0 if ok else 1)
