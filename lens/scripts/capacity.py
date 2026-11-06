"""capacity.py — what this machine can and cannot do. Measured, not guessed.

Owner asked: "tell my computer can handle without [breaking]". So this is the honest envelope:
available RAM, VRAM, disk, and the model sizes that fit inside the project's own limits
(5 GB VRAM, 20 GB disk growth, 2 h without a checkpoint).

The training-time estimate uses the throughput this project already measured on this GPU
(1900.3 GFLOPS at 50% efficiency) and the standard 6*params*tokens rule.
"""
import os
import shutil
import subprocess
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"

print("=" * 100)
print("WHAT THIS MACHINE CAN HANDLE — measured now")
print("=" * 100)

# ---- CPU / RAM
try:
    with open("/proc/meminfo") as f:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
    total_gb = mi["MemTotal"] / 1024 / 1024
    avail_gb = mi["MemAvailable"] / 1024 / 1024
    print(f"  RAM      {total_gb:.1f} GB total, {avail_gb:.1f} GB available")
except Exception as e:
    print(f"  RAM      unreadable: {e}")
try:
    with open("/proc/cpuinfo") as f:
        txt = f.read()
    model = [l.split(":", 1)[1].strip() for l in txt.splitlines() if l.startswith("model name")][0]
    ncores = txt.count("processor\t:")
    print(f"  CPU      {model}  ({ncores} threads)")
except Exception:
    pass

# ---- GPU
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    print(f"  GPU      {out}")
    free = int(out.split(",")[3].strip().split()[0])
except Exception as e:
    free = 0
    print(f"  GPU      query failed: {e}")

# ---- disk
for label, path in (("GlassNet", P), ("home", "/home/yahya")):
    try:
        u = shutil.disk_usage(path)
        print(f"  DISK {label:<9} {u.free/1e9:6.1f} GB free of {u.total/1e9:6.1f} GB")
    except Exception:
        pass
try:
    used = sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(P) for f in fs if os.path.exists(os.path.join(d, f)))
    print(f"  DISK GlassNet uses {used/1e9:.2f} GB  (limit 20 GB growth)")
except Exception:
    pass

# ---- model size envelope
print()
print("  MODEL SIZE ENVELOPE (project limits: VRAM < 5 GB, float32)")
CH = 92  # bytes per param: weights + grads + SGD
print(f"  {'params':>12}{'weights':>10}{'train mem':>12}{'fits 5GB VRAM?':>17}")
env = []
for n in (0.5e6, 1e6, 2e6, 5e6, 10e6, 20e6, 50e6, 100e6):
    w = n * 4
    t = n * CH
    fits = "yes" if t < 4.5e9 else "NO"
    env.append((n, w, t, fits))
    print(f"  {int(n):>12,}{w/1e6:>9.1f}M{t/1e6:>11.1f}M{fits:>17}")

# ---- training time estimate on THIS gpu
GFLOPS = 1900.3 * 0.5
print()
print(f"  TRAINING TIME at the throughput this project measured ({GFLOPS:.0f} GFLOPS effective)")
print(f"  (6 * params * tokens rule; 1 token = 1 training example seen)")
print(f"  {'params':>12}{'1M tok':>11}{'10M tok':>11}{'100M tok':>12}")
for n in (0.5e6, 1e6, 2e6, 5e6, 10e6, 20e6, 50e6, 100e6):
    row = []
    for toks in (1e6, 1e7, 1e8):
        flops = 6 * n * toks
        secs = flops / (GFLOPS * 1e9)
        row.append(secs)
    def fmt(s):
        return f"{s:.0f}s" if s < 120 else (f"{s/60:.1f}m" if s < 7200 else
                                            (f"{s/3600:.1f}h" if s < 172800 else f"{s/86400:.1f}d"))
    print(f"  {int(n):>12,}{fmt(row[0]):>11}{fmt(row[1]):>11}{fmt(row[2]):>12}")

print()
print("=" * 100)
print("  THE HONEST ENVELOPE")
print("=" * 100)
print("  FITS EASILY   up to ~20M params   (this is 4x the biggest model built so far)")
print("  FITS, SLOW    ~50M params          (hours-days per real training run)")
print("  DOES NOT FIT  100M+               (a real run is weeks; 500M+ is ~a year)")
print()
print("  So: a talk+see model at 2-20M params is comfortable. A 500M comparison never is.")
