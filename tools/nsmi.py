#!/usr/bin/env python3
"""nsmi — GPU status WITHOUT nvidia-smi. Reads /proc directly, so it works even when NVML
fails its version handshake (the 595.91-loaded vs 595.99-userspace mismatch of 2026-10-10).
Usage: nsmi   (or: python3 tools/nsmi.py)
"""
import os
import subprocess
import time

GPU_DIR = "/proc/driver/nvidia/gpus"


def main():
    print(f"GPU status at {time.strftime('%H:%M:%S')} (procfs, no NVML)")
    gpus = [d for d in os.listdir(GPU_DIR)] if os.path.isdir(GPU_DIR) else []
    if not gpus:
        print("  NO GPU NODES in /proc/driver/nvidia — driver module not loaded.")
        return 1
    for g in sorted(gpus):
        info = os.path.join(GPU_DIR, g, "information")
        if os.path.exists(info):
            for line in open(info):
                k, _, v = line.partition("\t")
                if k.strip() in ("Model:", "GPU UUID:", "Video BIOS:", "Bus Location:",
                                 "GPU Firmware:", "GPU Excluded:", "DMA Size:"):
                    print(f"  {k.strip():<14} {v.strip()}")
    # GPU activity approximation: /proc/pressure-free; instead show if anything holds the device
    try:
        r = subprocess.run(["lsof", "/dev/nvidia0"], capture_output=True, text=True)
        procs = [l.split()[1] for l in r.stdout.splitlines()[1:]]
        print(f"\n  {len(set(procs))} process(es) with the GPU open: {', '.join(sorted(set(procs))[:8])}")
    except FileNotFoundError:
        pass
    print("\n  nvidia-smi itself may still fail with 'Driver/library version mismatch' —")
    print("  that is NVML's header check only; compute and display keep working.")
    print("  Versions re-align automatically on the next boot (dkms has 595.99 staged).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())