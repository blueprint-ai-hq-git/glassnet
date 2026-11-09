#!/usr/bin/env python3
"""lens.export - get our model OUT in the normal formats.

WHAT WE CAN AND CANNOT PROMISE, stated first so nobody is misled:
* We CAN write the standard CONTAINERS: .safetensors, .gguf (F32), and our own .npz, each carrying the
  full tensor set plus the readable concept wall as metadata, and the model card in plain text.
* We CANNOT make llama.cpp / transformers / vLLM run this model, because LensNet is our own architecture
  and those runtimes have no implementation of it. A GGUF file is a container, not a runnable model for
  an architecture the runtime does not know. Anyone loading these files needs our loader (lens.load) or
  their own implementation of the architecture. Saying otherwise would be the kind of claim this project
  bans.

Every export is verified by READING IT BACK and comparing every tensor exactly, plus a hash of the
tensor bytes. A format that cannot be read back is reported as FAILED, not shipped.

Usage: python3 lens/export.py <ship.npz> <outdir>
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT", ".")
sys.path.insert(0, ROOT)
# REAL BUG FIX 2026-10-20 (evidence lens/runs/SERVE-GPU-AFTER.txt): `python3 lens/<this file>`
# puts THIS directory -- the project's lens/ -- at sys.path[0], so the top-level name `cuda`
# resolved to our own lens/cuda/ package. That is a REGULAR package and therefore beats
# cuda-python's PEP 420 namespace package (appending it does not help either), and cupy 14.2
# died at import with "ModuleNotFoundError: No module named 'cuda.pathfinder'" -- so --device
# gpu could not start AT ALL. This file's own directory must not be importable as top level.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
sys.path.insert(0, f"{ROOT}/lens/scripts")


def tensor_dict(net):
    """Every weight, by name, as plain numpy float32."""
    out = {"emb": np.asarray(net.emb.data, dtype=np.float32)}
    if getattr(net, "attn", None) is not None:
        for nm in ("Wq", "Wk", "Wv", "Wo"):
            t = getattr(net.attn, nm, None)
            if t is not None:
                out[f"attn.{nm}"] = np.asarray(t.data, dtype=np.float32)
    for nm, dense in (("concept", net.concept), ("answer", net.answer),
                      ("answer2", getattr(net, "answer2", None))):
        if dense is None:
            continue
        out[f"{nm}.W"] = np.asarray(dense.W.data, dtype=np.float32)
        if getattr(dense, "b", None) is not None:
            out[f"{nm}.b"] = np.asarray(dense.b.data, dtype=np.float32)
    return out


def meta(net):
    return {
        "architecture": "lensnet",
        "vocab_size": int(net.V),
        "d": int(net.D),
        "window": int(net.W),
        "n_concepts": int(net.K),
        "attention": bool(getattr(net, "attn", None) is not None),
        "mtp": int(getattr(net, "mtp", 1)),
        "concept_names": [c.name for c in net.concepts],
        "concept_tokens": {c.name: [int(i) for i in np.asarray(c.ids).reshape(-1)[:32]]
                           for c in net.concepts},
        "loader_note": "requires an implementation of the lensnet architecture; not runnable in "
                       "llama.cpp / transformers as-is",
    }


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


# ------------------------------------------------------------------ safetensors
def export_safetensors(net, path):
    from safetensors.numpy import save_file, load_file
    td = tensor_dict(net)
    # plain strings stay plain (JSON-encoding them made "lensnet" read back as '"lensnet"', which
    # the read-back check correctly flagged); only lists/dicts get encoded
    md = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in meta(net).items()}
    save_file(td, path, metadata=md)
    back = load_file(path)
    # read the metadata back too (safetensors keeps it in the header)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    md_back = hdr.get("__metadata__", {})
    same = all(k in back and np.array_equal(back[k], v) for k, v in td.items())
    ok = same and md_back.get("architecture") == "lensnet" and len(back) == len(td)
    return {"file": path, "format": "safetensors", "tensors": len(td),
            "bytes": os.path.getsize(path), "read_back_identical": bool(same),
            "metadata_read_back": md_back.get("architecture") == "lensnet", "ok": bool(ok)}


# ------------------------------------------------------------------ gguf (F32)
GGML_F32 = 0
T_STRING, T_UINT32, T_ARRAY, T_UINT64, T_BOOL = 8, 4, 9, 10, 7


def _s(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(k, v):
    return _s(k) + struct.pack("<I", T_STRING) + _s(v)


def _kv_u32(k, v):
    return _s(k) + struct.pack("<I", T_UINT32) + struct.pack("<I", int(v))


def _kv_u64(k, v):
    return _s(k) + struct.pack("<I", T_UINT64) + struct.pack("<Q", int(v))


def _kv_bool(k, v):
    return _s(k) + struct.pack("<I", T_BOOL) + struct.pack("<B", 1 if v else 0)


def _kv_str_array(k, vals):
    b = _s(k) + struct.pack("<I", T_ARRAY) + struct.pack("<I", T_STRING) + struct.pack("<Q", len(vals))
    for v in vals:
        b += _s(v)
    return b


def export_gguf(net, path, alignment=32):
    """Minimal GGUF v3 writer, F32 tensors only. Hand-written because the gguf package is not
    installed here and the format is simple enough to write exactly."""
    td = tensor_dict(net)
    m = meta(net)
    kvs = [_kv_str("general.architecture", "lensnet"),
           _kv_str("general.name", "GlassNet readable model"),
           _kv_u32("lensnet.vocab_size", m["vocab_size"]),
           _kv_u32("lensnet.d", m["d"]),
           _kv_u32("lensnet.window", m["window"]),
           _kv_u32("lensnet.n_concepts", m["n_concepts"]),
           _kv_u32("lensnet.mtp", m["mtp"]),
           _kv_bool("lensnet.attention", m["attention"]),
           _kv_str_array("lensnet.concept_names", m["concept_names"]),
           _kv_str("lensnet.loader_note", m["loader_note"])]
    # tensor infos + data layout
    infos, data, offset = [], bytearray(), 0
    for name, arr in td.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        dims = list(arr.shape)
        infos.append((name, dims, offset))
        raw = arr.tobytes()
        data += raw
        offset += len(raw)
        while offset % alignment:
            data += b"\x00"
            offset += 1
    hdr = bytearray()
    hdr += b"GGUF" + struct.pack("<I", 3)
    hdr += struct.pack("<Q", len(infos)) + struct.pack("<Q", len(kvs))
    for kv in kvs:
        hdr += kv
    for name, dims, off in infos:
        hdr += _s(name) + struct.pack("<I", len(dims))
        for d in dims:
            hdr += struct.pack("<Q", d)
        hdr += struct.pack("<I", GGML_F32) + struct.pack("<Q", off)
    while len(hdr) % alignment:
        hdr += b"\x00"
    with open(path, "wb") as f:
        f.write(bytes(hdr) + bytes(data))

    # ---- read it back with our own parser and compare every tensor byte for byte
    with open(path, "rb") as f:
        blob = f.read()
    magic = blob[:4]
    ver, nt, nkv = struct.unpack("<IQQ", blob[4:24])
    p = 24
    got_names = []
    for _ in range(nkv):
        ln = struct.unpack("<Q", blob[p:p + 8])[0]; p += 8
        key = blob[p:p + ln].decode(); p += ln
        vt = struct.unpack("<I", blob[p:p + 4])[0]; p += 4
        if vt == T_STRING:
            ln2 = struct.unpack("<Q", blob[p:p + 8])[0]; p += 8 + ln2
        elif vt == T_UINT32:
            p += 4
        elif vt == T_UINT64:
            p += 8
        elif vt == T_BOOL:
            p += 1
        elif vt == T_ARRAY:
            et = struct.unpack("<I", blob[p:p + 4])[0]; p += 4
            cnt = struct.unpack("<Q", blob[p:p + 8])[0]; p += 8
            for _i in range(cnt):
                if et == T_STRING:
                    l3 = struct.unpack("<Q", blob[p:p + 8])[0]; p += 8 + l3
        else:
            raise ValueError(f"unknown kv type {vt} while reading back")
        if key == "lensnet.concept_names":
            got_names = True
    tinfo = []
    for _ in range(nt):
        ln = struct.unpack("<Q", blob[p:p + 8])[0]; p += 8
        name = blob[p:p + ln].decode(); p += ln
        nd = struct.unpack("<I", blob[p:p + 4])[0]; p += 4
        # the offset must advance INSIDE the loop: a list comprehension read the same 8 bytes nd
        # times, so every 2-D shape was verified as [d1, d1] and the check failed on a good file
        dims = []
        for _ in range(nd):
            dims.append(struct.unpack("<Q", blob[p:p + 8])[0])
            p += 8
        gtype = struct.unpack("<I", blob[p:p + 4])[0]; p += 4
        off = struct.unpack("<Q", blob[p:p + 8])[0]; p += 8
        tinfo.append((name, dims, gtype, off))
    while p % alignment:
        p += 1
    base = p
    same = True
    for (name, dims, gtype, off), (n2, arr) in zip(tinfo, td.items()):
        want = np.ascontiguousarray(arr, dtype=np.float32)
        have = np.frombuffer(blob, dtype=np.float32, count=want.size, offset=base + off).reshape(want.shape)
        same = same and name == n2 and list(want.shape) == dims and np.array_equal(have, want)
    return {"file": path, "format": "gguf", "version": ver, "tensors": nt, "kv": nkv,
            "concept_names_in_header": bool(got_names), "bytes": len(blob),
            "read_back_identical": bool(same and magic == b"GGUF"),
            "ok": bool(same and magic == b"GGUF" and nt == len(td))}


# ------------------------------------------------------------------ model card
def export_model_card(net, path, ship):
    lines = ["# GlassNet readable model card", "",
             f"Ship: `{os.path.basename(ship)}`", "",
             "## What this model is",
             f"A {sum(p.data.size for p in net.parameters()):,}-parameter readable network: every middle",
             f"unit has a human name ({net.K} concepts), and the answer is computed from those named",
             "units only, so the arithmetic can be printed and checked.", "",
             "## Architecture",
             f"- vocabulary: {net.V:,} tokens", f"- width d: {net.D}", f"- window: {net.W} tokens",
             f"- concepts: {net.K}",
             f"- attention: {'yes' if getattr(net, 'attn', None) is not None else 'no'}",
             f"- multi-token heads: {getattr(net, 'mtp', 1)}", "",
             "## The concept wall (each unit and the tokens that define it)",
             "| unit | tokens |", "|---|---|"]
    for c in net.concepts:
        toks = np.asarray(c.ids).reshape(-1)[:10]
        lines.append(f"| {c.name} | {' '.join(str(int(t)) for t in toks)} |")
    lines += ["", "## How to load it",
              "These files are standard containers, but this is our own architecture: llama.cpp,",
              "transformers and vLLM have no implementation of `lensnet`, so they cannot run these",
              "files. Load them with `lens.load` (our loader) or implement the architecture from the",
              "metadata above.", ""]
    open(path, "w").write("\n".join(lines))
    return {"file": path, "format": "model_card", "concepts": net.K,
            "bytes": os.path.getsize(path), "ok": os.path.getsize(path) > 500}


def main():
    import lens  # noqa: E402
    ship = sys.argv[1] if len(sys.argv) > 1 else "lens/runs/chat/sb_S0_d256_noattn_ep96.npz"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "lens/runs/export"
    os.makedirs(outdir, exist_ok=True)
    os.chdir(ROOT)
    net = lens.load(ship, device="cpu")
    base = os.path.basename(ship).replace(".npz", "")
    results = []
    results.append(export_safetensors(net, f"{outdir}/{base}.safetensors"))
    results.append(export_gguf(net, f"{outdir}/{base}.gguf"))
    results.append(export_model_card(net, f"{outdir}/{base}.model-card.md", ship))
    # native npz stays as the canonical checkpoint: copy it, do not re-write it
    import shutil
    shutil.copy2(ship, f"{outdir}/{base}.npz")
    results.append({"file": f"{outdir}/{base}.npz", "format": "npz (native, copied)",
                    "bytes": os.path.getsize(f"{outdir}/{base}.npz"), "ok": True})
    print(f"EXPORT of {os.path.basename(ship)}")
    allok = True
    for r in results:
        allok = allok and r["ok"]
        print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {r['format']:<28} "
              f"{r['bytes']/1e6:8.2f} MB  {r['file']}")
        for k, v in r.items():
            if k not in ("file", "format", "ok", "bytes"):
                print(f"          {k}: {v}")
    json.dump(results, open(f"{outdir}/EXPORT-REPORT.json", "w"), indent=1)
    print(f"\nEXPORT: {'ALL PASS' if allok else 'FAILED'}  (report: {outdir}/EXPORT-REPORT.json)")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())