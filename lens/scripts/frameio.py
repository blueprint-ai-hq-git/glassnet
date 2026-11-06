"""frameio.py — draw on frames and write real video, with NO image libraries.

Nothing here imports PIL, cv2 or imageio -- none are installed and none are on this project's
allowed list. Everything is done with the standard library plus ffmpeg, which is already on this
machine:

    * `FONT`      a 5x7 bitmap font, A-Z 0-9 and punctuation, written out by hand
    * `draw_text` renders a string into an RGB frame with the font
    * `draw_rect` / `draw_bar` for boxes and confidence bars
    * `VideoWriter` pipes raw rgb24 frames into ffmpeg and produces a real .mp4

This is what makes the "virtual camera" possible: frames in, the model's own answer drawn on them,
video out -- no physical camera touched anywhere.
"""
import os
import shutil
import subprocess

import numpy as np

# ------------------------------------------------------------------------------------------------
# 5x7 bitmap font. Each glyph is 7 rows of 5 pixels, '#' = on.
# ------------------------------------------------------------------------------------------------
FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "11110", "10001", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "11110", "10000", "10000", "10000", "11111"],
    "F": ["11111", "10000", "11110", "10000", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "#": ["01010", "01010", "11111", "01010", "11111", "01010", "01010"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    "?": ["01110", "10001", "00001", "00110", "00100", "00000", "00100"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    ",": ["00000", "00000", "00000", "00000", "00110", "00100", "01000"],
    "<": ["00010", "00100", "01000", "10000", "01000", "00100", "00010"],
    ">": ["01000", "00100", "00010", "00001", "00010", "00100", "01000"],
    "*": ["00000", "10101", "01110", "11111", "01110", "10101", "00000"],
}


def text_width(s, scale=2):
    return len(s) * 6 * scale


def draw_text(frame, s, x, y, scale=2, colour=(255, 255, 255), bg=None):
    """Draw a string using the 5x7 font. Unknown characters render as blanks, never raise."""
    s = str(s).upper()
    if bg is not None:
        w = text_width(s, scale) + 2 * scale
        h = 7 * scale + 2 * scale
        draw_rect(frame, x - scale, y - scale, w, h, bg)
    for ci, ch in enumerate(s):
        glyph = FONT.get(ch)
        if glyph is None:
            continue
        for ry, row in enumerate(glyph):
            for rx, px in enumerate(row):
                if px != "1":
                    continue
                y0 = y + ry * scale
                x0 = x + (ci * 6) * scale + rx * scale
                frame[y0:y0 + scale, x0:x0 + scale] = colour
    return frame


def draw_rect(frame, x, y, w, h, colour, thick=1):
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return frame
    if thick <= 1:
        frame[y0:y1, x0:x1] = colour
    else:
        frame[y0:y0 + thick, x0:x1] = colour
        frame[y1 - thick:y1, x0:x1] = colour
        frame[y0:y1, x0:x0 + thick] = colour
        frame[y0:y1, x1 - thick:x1] = colour
    return frame


def draw_bar(frame, x, y, w, h, frac, colour=(0, 255, 0), bg=(40, 40, 40)):
    draw_rect(frame, x, y, w, h, bg)
    fw = int(max(0.0, min(1.0, float(frac))) * w)
    if fw > 0:
        draw_rect(frame, x, y, fw, h, colour)
    return frame


def upscale(img28, size):
    """Nearest-neighbour upscale of a 28x28 float image to (size, size) uint8."""
    a = np.asarray(img28, dtype=np.float32).reshape(28, 28)
    a = np.clip(a, 0.0, 1.0)
    step = 28.0 / size
    idx = np.clip((np.arange(size) * step).astype(int), 0, 27)
    small = (a * 255.0).astype(np.uint8)[np.ix_(idx, idx)]
    return np.stack([small] * 3, axis=-1)


class VideoWriter:
    """Pipe raw rgb24 frames into ffmpeg and get a real .mp4 out. No image library involved."""

    def __init__(self, path, width, height, fps=10, crf=23):
        self.path = path
        self.w, self.h, self.fps = width, height, fps
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ff = shutil.which("ffmpeg") or "/home/yahya/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"
        self.ffmpeg = ff
        cmd = [ff, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        self.n = 0

    def write(self, frame):
        f = np.ascontiguousarray(frame, dtype=np.uint8)
        if f.shape[0] != self.h or f.shape[1] != self.w:
            raise ValueError(f"frame is {f.shape}, writer is {self.h}x{self.w}")
        self.proc.stdin.write(f.tobytes())
        self.n += 1

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        err = self.proc.stderr.read().decode("utf-8", "replace")
        rc = self.proc.wait()
        if rc != 0:
            return False, err.strip()[:400]
        return True, f"{self.n} frames -> {self.path}"


def has_ffmpeg():
    return bool(shutil.which("ffmpeg")
                or os.path.exists("/home/yahya/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"))
