#!/bin/bash
# find_nvrtc.sh — locate a usable libnvrtc for our own runtime kernel compilation.
echo "=== candidate libnvrtc files ==="
find / -name "libnvrtc.so*" -not -path "*/proc/*" 2>/dev/null | head -10
echo
echo "=== can ctypes load any of them? ==="
python3 - <<'EOF'
import ctypes, glob, os
cands = []
for root in ("/run/media/yahya/work", "/usr/lib", "/usr/local", os.path.expanduser("~")):
    cands += glob.glob(os.path.join(root, "**", "libnvrtc.so*"), recursive=True)
cands = [c for c in cands if os.path.isfile(c)][:12]
for c in cands:
    try:
        lib = ctypes.CDLL(c)
        fn = lib.nvrtcVersion
        fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        major, minor = ctypes.c_int(), ctypes.c_int()
        rc = fn(ctypes.byref(major), ctypes.byref(minor))
        print(f"  OK   {c}  -> nvrtcVersion rc={rc} version {major.value}.{minor.value}")
    except Exception as e:
        print(f"  no   {c}  ({type(e).__name__})")
EOF
echo
echo "=== does cupy ship one? ==="
ls /run/media/yahya/HARD\ DISK\ 4/GlassNet/.venv-gpu/lib/python3.11/site-packages/cupy*/ 2>/dev/null | head -5
find "/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu" -name "*nvrtc*" 2>/dev/null | head -5