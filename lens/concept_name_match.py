"""NAME THE UNNAMED CONCEPTS BY MEASUREMENT, NOT BY GUESSING.

Four of the 14 concept columns in data/concepts.npz have no name in data/concepts_disp10.json - they appear as unnamed_10..13.
Two of them (unnamed_11 and unnamed_13) are among the SIX concepts a reader must actually read, so an unnamed concept is a direct
hole in this project's one promise: that every middle neuron has a human-readable meaning.

Instead of inventing names, the stored columns are matched against the NAMED definitions in glassnet/concept_pool.py by recomputing
that pool from the same images and correlating. A column that matches a named definition at |r| ~ 1.0 IS that definition, and can
carry its name with evidence. A column that matches nothing gets no name and is reported as still unknown.

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 every one of the 14 stored columns matches some pool definition at |r| >= 0.99, so all 14 can be named with evidence.
FALSIFIER: any column that matches nothing above 0.99 stays unnamed and is reported as such - a name is never invented to fill a hole.
"""
import gzip
import json
import os
import struct
import sys

import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT", os.environ.get("GLASSNET_ROOT", "."))
sys.path.insert(0, ROOT)
from glassnet.concept_pool import compute_pool, DESC


def read_idx(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def main():
    z = np.load(os.path.join(ROOT, "data", "concepts.npz"))
    Cte = z["Cte"].astype(np.float64)
    names = list(json.load(open(os.path.join(ROOT, "data", "concepts_disp10.json")))["names"])
    while len(names) < Cte.shape[1]:
        names.append("unnamed_%d" % len(names))

    img = read_idx(os.path.join(ROOT, "data", "t10k-images-idx3-ubyte.gz"))[: len(Cte)]
    X = img.reshape(len(img), -1).astype(np.float32) / 255.0
    pool, pool_names = compute_pool(X)
    print("=== naming the unnamed concepts by measurement, %s ===" % __import__("time").strftime("%Y-%m-%d %H:%M:%S"))
    print("stored columns %d, pool definitions available %d" % (Cte.shape[1], len(pool_names)))
    print()
    print("  %-16s %-22s %8s   %s" % ("stored column", "matches definition", "corr", "description"))
    unknown = []
    mapping = {}
    for j in range(Cte.shape[1]):
        cors = np.array([corr(Cte[:, j], pool[:, i]) for i in range(pool.shape[1])])
        best = int(np.argmax(np.abs(cors)))
        r = float(cors[best])
        mapping[j] = (pool_names[best], abs(r))
        desc = str(DESC.get(pool_names[best], ""))[:60]
        flag = "" if abs(r) >= 0.99 else "   <- NO CONFIDENT MATCH"
        print("  %-16s %-22s %8.4f   %s%s" % (names[j], pool_names[best], r, desc, flag))
        if abs(r) < 0.99:
            unknown.append(names[j])
    print()
    named = Cte.shape[1] - len(unknown)
    print("RESULT %d of %d stored columns match a named definition at |r| >= 0.99" % (named, Cte.shape[1]))
    print("P1 all 14 matched: %s" % ("MET" if not unknown else "NOT MET"))
    if unknown:
        print("STILL UNNAMED (no name invented): %s" % ", ".join(unknown))
        print("OUTCOME: SOME_CONCEPTS_UNNAMED")
    else:
        print("OUTCOME: ALL_CONCEPTS_NAMED")
    print()
    print("--- the corrected concept list, with the definition each name now stands on ---")
    for j in range(Cte.shape[1]):
        print("  %2d  %-16s = %s" % (j, names[j] if not names[j].startswith("unnamed") else "unnamed_%d" % j,
                                     mapping[j][0]))
    json.dump({str(j): {"stored": names[j], "definition": mapping[j][0], "corr": mapping[j][1]}
               for j in range(Cte.shape[1])},
              open(os.path.join(ROOT, "runs", "verify", "concept-name-map.json"), "w"), indent=1)
    print("map written to runs/verify/concept-name-map.json")
    print("CONCEPT-NAME-MATCH COMPLETE")


if __name__ == "__main__":
    main()
