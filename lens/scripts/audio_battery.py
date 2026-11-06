"""audio_battery.py — the SCENARIO BATTERY that caught BUG-A1, re-run against the fixed runtime.

BUG-A1 was: the runtime answered has_voiced (1.0) for a 440 Hz tone, a 120 Hz tone and white
noise. The trained concepts were honest (band-shape / level formulas) but the ANSWER overclaimed
the word "voiced". The fix composes the verdict at the runtime from the trained concepts PLUS two
hand-computed statistics (spectral flatness, peak-to-mean), each printed with its threshold.

GATE (pre-registered): all four scenarios answered correctly --
    real voice recording -> voiced
    440 Hz tone          -> tonal
    white noise          -> noisy
    near-silent file     -> silence
Any scenario that lands elsewhere is a named failure, printed with its statistics.
"""
import os
import sys
import wave

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import variants as V          # noqa: E402

OUT = os.path.join(PROJ, "lens", "runs", "audio_battery")
os.makedirs(OUT, exist_ok=True)
SR = 16000


def write_wav(path, x):
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())
    return path


def main():
    rng = np.random.default_rng(0)
    t = np.arange(SR * 3) / SR
    scenarios = [
        ("voice (real recording)", "/home/yahya/amar_slang_p1.ogg", "voiced"),
        ("440 Hz tone", write_wav(os.path.join(OUT, "tone440.wav"),
                                  0.3 * np.sin(2 * np.pi * 440 * t)), "tonal"),
        ("white noise", write_wav(os.path.join(OUT, "noise.wav"),
                                  rng.normal(0, 0.3, len(t))), "noisy"),
        ("near-silent", write_wav(os.path.join(OUT, "silent.wav"),
                                  rng.normal(0, 1e-4, len(t))), "silence"),
    ]
    vr = V.build_default()
    from lens import audio as A
    fs = [os.path.join(PROJ, "lens", "runs", "audio", f)
          for f in os.listdir(os.path.join(PROJ, "lens", "runs", "audio")) if f.endswith(".npz")]
    newest = max(fs, key=os.path.getmtime)
    vr.models["audio"] = (A.load(newest, device="cpu"), None)
    print(f"audio ship: {os.path.basename(newest)}")

    ok = 0
    for name, path, expect in scenarios:
        r = vr.listen(path)
        got = r.get("answer")
        good = got == expect
        ok += good
        print(f"\n  {name:<24} -> {got:<8} expected {expect:<8} "
              f"[{'OK' if good else 'WRONG'}]")
        print(f"      stats {r.get('stats')}  composition: {r.get('composition')}")
        print(f"      concepts: {r.get('fired_concepts')}")
    print(f"\n  AUDIO BATTERY: {ok}/{len(scenarios)} scenarios correct "
          f"-> {'PASS' if ok == len(scenarios) else 'FAIL'}")
    return 0 if ok == len(scenarios) else 2


if __name__ == "__main__":
    sys.exit(main())