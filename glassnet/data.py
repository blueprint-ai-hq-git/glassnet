"""glassnet.data — Fashion-MNIST (MIT licensed) download + load.

Downloads the four official gz files from the zalando-research repo, verifies
their parsed shapes and label ranges (that IS the integrity check), and caches
a re-normalised .npz so we never re-download. Pixels: float32, [0, 1].
"""
import gzip
import os
import struct
import urllib.request

import numpy as np

_BASE = ("https://raw.githubusercontent.com/zalandoresearch/"
         "fashion-mnist/master/data/fashion")
_FILES = {
    "train-images-idx3-ubyte.gz": (60000, 28, 28),
    "train-labels-idx1-ubyte.gz": (60000,),
    "t10k-images-idx3-ubyte.gz": (10000, 28, 28),
    "t10k-labels-idx1-ubyte.gz": (10000,),
}
LABELS = ["t-shirt/top", "trouser", "pullover", "dress", "coat",
          "sandal", "shirt", "sneaker", "bag", "ankle boot"]


def _download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for fname in _FILES:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        url = f"{_BASE}/{fname}"
        print(f"[data] downloading {fname} ...", flush=True)
        urllib.request.urlretrieve(url, path)


def _read_idx(path, shape):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        n_dims = magic & 0xFF
        dims = struct.unpack(">" + "I" * n_dims, f.read(4 * n_dims))
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(dims)
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"{path}: parsed shape {arr.shape} != expected {shape}")
    return arr


def load(data_dir, force=False):
    """Returns (X_train, y_train, X_test, y_test)."""
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, "fashion_mnist_cache.npz")
    if not force and os.path.exists(cache):
        z = np.load(cache)
        return z["X_train"], z["y_train"], z["X_test"], z["y_test"]

    _download(data_dir)
    Xtr = _read_idx(os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                    (60000, 28, 28))
    ytr = _read_idx(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
                    (60000,))
    Xte = _read_idx(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
                    (10000, 28, 28))
    yte = _read_idx(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
                    (10000,))
    for a in (ytr, yte):
        if set(np.unique(a)) != set(range(10)):
            raise ValueError("labels outside 0..9 — corrupt download, refusing")

    X_train = Xtr.reshape(60000, 784).astype(np.float32) / 255.0
    X_test = Xte.reshape(10000, 784).astype(np.float32) / 255.0
    np.savez(cache, X_train=X_train, y_train=ytr, X_test=X_test, y_test=yte)
    print(f"[data] cached -> {cache}", flush=True)
    return X_train, ytr, X_test, yte