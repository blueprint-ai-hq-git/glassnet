
import os, sys, glob
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet")
import lens.api as api
ships = sorted(glob.glob("/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat/*.npz"),
               key=os.path.getmtime, reverse=True)
found = []
for s in ships[:60]:
    try:
        net = api.load(s, device="cpu")
        if getattr(net, "attn", None) is not None:
            found.append((os.path.basename(s), net.D, net.W, os.path.getmtime(s)))
    except Exception:
        continue
print("ships WITH attention (newest first):")
for f in found[:6]:
    print("  ", f[0], "d=%d w=%d" % (f[1], f[2]))
if not found:
    print("   none found in the 60 newest; the cache needs an attention ship to test against")
