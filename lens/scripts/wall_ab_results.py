#!/usr/bin/env python3
"""Parse lens/runs/WALL-AB.txt into lens/runs/WALL-AB-RESULTS.tsv.

Only real numbers from the log are copied - nothing is computed or estimated.
Segments are delimited by the runner's own marker lines:

    ----- <arm> seed <S> : <cmd> <HH:MM:SS> vram_free=<N>MiB
    ...
      final VAL ppl <ppl>  acc <acc>%  concepts <k>/<n>

The LAST final VAL line inside a segment wins (a segment can contain a resumed
or re-run arm). Segments without a final VAL line are dropped, so a killed arm
never appears as a result.
"""
import re
import sys
from pathlib import Path

ROOT = Path("/run/media/yahya/HARD DISK 4/GlassNet")
LOG = ROOT / "lens/runs/WALL-AB.txt"
OUT = ROOT / "lens/runs/WALL-AB-RESULTS.tsv"

MARK = re.compile(r"^----- (\S+) seed (\d+) :")
VAL = re.compile(r"final VAL ppl ([\d.]+)\s+acc ([\d.]+)%\s+concepts (\d+/\d+)")


def main() -> int:
    if not LOG.exists():
        print("no log yet", file=sys.stderr)
        return 1
    rows: dict[str, dict] = {}
    arm = seed = None
    ok = 0
    for n, line in enumerate(LOG.read_text(errors="replace").splitlines(), 1):
        m = MARK.match(line)
        if m:
            arm, seed = m.group(1), int(m.group(2))
            continue
        v = VAL.search(line)
        if v and arm is not None:
            tag = f"wall_{arm}_s{seed}"
            rows[tag] = {
                "tag": tag,
                "arm": arm,
                "seed": str(seed),
                "ppl": v.group(1),
                "acc": v.group(2),
                "concepts": v.group(3),
                "src": f"lens/runs/WALL-AB.txt:{n}",
            }
            ok += 1
    if not rows:
        print("no finished arm yet", file=sys.stderr)
        return 1
    OUT.write_text("tag\tarm\tseed\tfinal_ppl\tfinal_acc_pct\tconcepts\tsource\n"
                   + "".join("\t".join([r["tag"], r["arm"], r["seed"], r["ppl"],
                                        r["acc"], r["concepts"], r["src"]]) + "\n"
                             for r in sorted(rows.values(), key=lambda r: (r["seed"], r["arm"]))))
    print(f"wrote {OUT} with {len(rows)} arm(s)")
    for r in sorted(rows.values(), key=lambda r: (r["seed"], r["arm"])):
        print(f"  {r['tag']:14s} ppl {r['ppl']:>8s}  acc {r['acc']:>6s}%  {r['src']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
