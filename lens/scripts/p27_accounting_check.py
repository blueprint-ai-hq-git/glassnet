"""Scratch: verify the P27 accounting claim - is the FAIL about memory, or about the driver's chunk loop?"""
import json

W = 32; STRIDE = 32; CHUNK = 32768; K = 36
rows = json.load(open("/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/warm_memory_stream.json"))["rows"]


def windows_for(target, pos):
    tot = 0; covered = 0
    for s in range(pos, target - W, CHUNK):
        ln = min(s + CHUNK, target - W) - s
        n = (ln - W) // STRIDE
        if n <= 0:
            continue
        tot += n; covered += n * STRIDE
    return tot, covered


pos = 0
for r in rows:
    t = r["tokens"]
    exp, cov = windows_for(t, pos)
    pos = t
    print(f"target {t:>12,}  stored {r['windows']:>10,}  driver-loop {exp:>10,}  diff {r['windows']-exp:>6}"
          f"  covered {cov:>12,}  tokens-not-in-any-window {t-cov:>8,}"
          f"  bytes-windows*K {r['bytes']-r['windows']*K:>4}"
          f"  bytes-arith {r['bytes']-r['arith_bytes']:>12,}")
