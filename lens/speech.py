"""SPEECH — the readable spoken-word recognizer as a first-class line of the system.

This module is the SINGLE home of the speech feature formulas and the trained recognizer. Both the
dataset builder (lens/scripts/speech_data.py) and the runtime import from here, so there is exactly
one implementation of every formula -- duplicated feature maths is how a "readable" claim quietly
stops being true.

Features (all hand-computable; see FORMULAS):
  dur_s, rms, zcr, centroid_hz, low_ratio, high_band_present->high_ratio, voiced_frac, flatness,
  peak_pos

Concepts (each = a threshold on ONE named feature, direction recorded):
  loud, noisy, bright, low_band_dominant, high_band_present, voiced, tonal, early_peak
  (`long_sound` is absent on purpose: every Speech Commands clip is exactly 1.00 s, so its label
  was constant -- the concept was dropped on evidence, recorded in LOG.md.)

Ship format: lens/runs/speech/speech_readable_s0.npz (W1,b1,W2,b2,W3,b3, concepts, words, feats),
pruned so every rule reads at most 3 concepts.
"""
import json
import os
import wave

import numpy as np

PROJ = os.environ.get("GLASSNET_ROOT", ".")
SR = 16000
FRAME = 400       # 25 ms
HOP = 160         # 10 ms
FEATS = ["dur_s", "rms", "zcr", "centroid_hz", "low_ratio", "high_ratio", "voiced_frac",
         "flatness", "peak_pos"]
FORMULAS = {
    "dur_s": "len(samples)/16000",
    "rms": "mean over 25ms/10ms frames of sqrt(mean(x^2))",
    "zcr": "mean over frames of (#sign changes)/len(frame)",
    "centroid_hz": "mean over frames of sum(f*P)/sum(P), f = rfft bin freq (Hz)",
    "low_ratio": "sum(P[<500Hz])/sum(P)",
    "high_ratio": "sum(P[>2000Hz])/sum(P)",
    "voiced_frac": "fraction of frames whose autocorrelation peak (lag 40..320 samples = 50..400Hz) "
                   "exceeds 0.5*autocorr[0] (computed by FFT, Wiener-Khinchin)",
    "flatness": "mean over frames of exp(mean(log(P+1e-12)))/mean(P+1e-12)",
    "peak_pos": "argmax over frames of frame energy / (#frames)",
}


def read_wav(path):
    """16-bit PCM WAV -> float mono at 16 kHz (stdlib `wave` only)."""
    with wave.open(path, "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"expected 16-bit PCM, got {sw*8}-bit")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr))
        x = np.interp(idx, np.arange(len(x)), x)
    return x / 32768.0


def features(x):
    """The nine documented formulas. Input: float samples at 16 kHz."""
    n = len(x)
    dur = n / SR
    if n < FRAME:
        x = np.pad(x, (0, FRAME - n))
        n = len(x)
    nf = 1 + (n - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(nf)[:, None]
    frames = x[idx] * np.hanning(FRAME)[None, :]
    rms = float(np.mean(np.sqrt(np.mean(frames ** 2, axis=1))))
    sgn_change = np.abs(np.diff(np.sign(frames), axis=1)) > 0
    zcr = float(np.mean(sgn_change.sum(axis=1) / FRAME))
    P = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, d=1.0 / SR)
    tot = P.sum(axis=1) + 1e-12
    centroid = float(np.mean((P * freqs[None, :]).sum(axis=1) / tot))
    low = float(P[:, freqs < 500].sum() / tot.sum())
    high = float(P[:, freqs > 2000].sum() / tot.sum())
    nfft = 2 * FRAME
    F = np.fft.rfft(frames, n=nfft, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, n=nfft, axis=1)[:, :FRAME]
    lags = np.arange(int(SR / 400), int(SR / 50))
    ac0 = ac[:, :1] + 1e-12
    voiced = float(np.mean((ac[:, lags].max(axis=1, keepdims=True) / ac0).ravel() > 0.5))
    gm = np.exp(np.mean(np.log(P + 1e-12), axis=1))
    am = np.mean(P + 1e-12, axis=1)
    flat = float(np.mean(gm / am))
    peak_pos = float(np.argmax((frames ** 2).mean(axis=1)) / max(nf - 1, 1))
    return [dur, rms, zcr, centroid, low, high, voiced, flat, peak_pos]


class WordNet:
    """The trained readable recognizer: features -> concepts (named) -> sparse rules -> words."""

    def __init__(self, W1, b1, W2, b2, W3, b3, concepts, words, feats, mu=None, sd=None):
        self.W1, self.b1 = np.asarray(W1), np.asarray(b1)
        self.W2, self.b2 = np.asarray(W2), np.asarray(b2)
        self.W3, self.b3 = np.asarray(W3), np.asarray(b3)
        self.concepts = [str(c) for c in concepts]
        self.words = [str(w) for w in words]
        self.feats = [str(f) for f in feats]
        self.mu, self.sd = mu, sd            # TRAIN-only standardisation (saved with the ship)

    def forward(self, X):
        Z = X if self.mu is None else (np.asarray(X) - self.mu) / self.sd
        h = 1 / (1 + np.exp(-np.clip(np.asarray(Z) @ self.W1 + self.b1, -60, 60)))
        pre = h @ self.W2 + self.b2
        r = np.maximum(pre, 0)
        return h, r, r @ self.W3 + self.b3

    def predict(self, x_samples):
        """Return (word, probability, top concept activations) for one clip."""
        f = np.asarray(features(x_samples), dtype=np.float32)[None, :]
        h, r, lg = self.forward(f)
        lg = lg[0] - lg[0].max()
        p = np.exp(lg) / np.exp(lg).sum()
        k = int(p.argmax())
        top = sorted(zip(self.concepts, h[0]), key=lambda kv: -kv[1])[:3]
        return self.words[k], float(p[k]), top


def load(path, device="cpu"):
    """Load a speech ship. `device` is accepted for API symmetry; the model is tiny and CPU-bound."""
    z = np.load(path, allow_pickle=True)
    mu = z["mu"] if "mu" in z.files else None
    sd = z["sd"] if "sd" in z.files else None
    return WordNet(z["W1"], z["b1"], z["W2"], z["b2"], z["W3"], z["b3"],
                   z["concepts"], z["words"], z["feats"], mu, sd)


def explain(net, x_samples, top_k=3):
    """The arithmetic behind one prediction, with names and numbers."""
    word, prob, tops = net.predict(x_samples)
    f = np.asarray(features(x_samples), dtype=np.float32)
    Z = f if net.mu is None else (f - net.mu) / net.sd
    h, r, lg = net.forward(f[None, :])
    rules = sorted(range(len(r[0])), key=lambda i: -r[0][i])[:top_k]
    lines = [f"features: " + ", ".join(f"{n}={v:.3g}" for n, v in zip(net.feats, f)),
             f"concepts: " + ", ".join(f"{n}={v:.3f}" for n, v in tops),
             f"decision: '{word}' p={prob:.3f}"]
    for rr in rules:
        reads = [(net.concepts[k], float(net.W2[k, rr])) for k in
                 np.argsort(-np.abs(net.W2[:, rr]))[:3] if net.W2[k, rr] != 0]
        lines.append(f"  rule #{rr} (act {r[0][rr]:.3f}) reads " +
                     ", ".join(f"{n}(w={w:+.2f})" for n, w in reads))
    return "\n".join(lines)
