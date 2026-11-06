#!/bin/bash
# check_nvrtc.sh — find and load a usable libnvrtc (targeted paths only; a full-disk find
# took 7 minutes and timed out).
echo "=== cupy's bundled libs ==="
ls "/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/lib/python3.11/site-packages/cupy/lib/" 2>/dev/null | head

echo
echo "=== candidate nvrtc paths ==="
for p in \
  "/run/media/yahya/work/onnx-venv/lib/python3.11/site-packages/nvidia/cu13/lib/libnvrtc.so.13" \
  "/run/media/yahya/work/1.5b-finetune/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12" \
  "/usr/lib/x86_64-linux-gnu/libnvrtc.so" \
  ; do
  if [ -f "$p" ]; then echo "  EXISTS $p"; else echo "  absent $p"; fi
done

echo
echo "=== load test ==="
python3 - <<'PYEOF'
import ctypes, os
cands = [
 "/run/media/yahya/work/onnx-venv/lib/python3.11/site-packages/nvidia/cu13/lib/libnvrtc.so.13",
 "/run/media/yahya/work/1.5b-finetune/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
 "/usr/lib/x86_64-linux-gnu/libnvrtc.so",
]
ok = None
for p in cands:
    if not os.path.exists(p):
        continue
    try:
        lib = ctypes.CDLL(p)
        f = lib.nvrtcVersion
        f.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        a, b = ctypes.c_int(), ctypes.c_int()
        rc = f(ctypes.byref(a), ctypes.byref(b))
        print(f"  OK   {p}")
        print(f"       nvrtcVersion rc={rc} -> {a.value}.{b.value}")
        ok = p
        break
    except Exception as e:
        print(f"  no   {p}  ({type(e).__name__}: {e})")
print(f"\nUSE: {ok}" if ok else "\nNO usable libnvrtc found on the targeted paths")
PYEOF