"""Ask the readable vision model why it answered - in named concepts you can recompute yourself.

Every number below is the model's own arithmetic: logits = h @ answer.W + answer.b, so each term h_k * W[k, class] is exactly what one named concept contributed.
Nothing here is an approximation or an attribution method - it IS the forward pass, rearranged.

    pip install numpy safetensors
    python examples/explain_image.py            # uses a syntactically generated test image
"""
import numpy as np

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load(model_dir=".", npz="model.safetensors"):
    from safetensors.numpy import load_file
    import json, os
    t = load_file(os.path.join(model_dir, npz))
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    return t, cfg


def forward(t, cfg, image):
    """patches -> shared patch projection -> named concept layer -> linear answer. Plain numpy, ~15 lines."""
    d, patch = cfg["d"], cfg["patch"]
    n = 28 // patch
    x = np.asarray(image, dtype=np.float32).reshape(28, 28)
    patches = np.stack([x[i * patch:(i + 1) * patch, j * patch:(j + 1) * patch].ravel() for i in range(n) for j in range(n)])
    feats = np.tanh(patches @ t["patch_proj.W"] + t["patch_proj.b"])          # (n*n, d)
    h = 1.0 / (1.0 + np.exp(-(feats.ravel() @ t["concept.W"] + t["concept.b"])))  # (K,) named concepts
    logits = h @ t["answer.W"] + t["answer.b"]
    return h, logits


def explain(t, cfg, image, top_k=4):
    h, logits = forward(t, cfg, image)
    pred = int(logits.argmax())
    contrib = h * t["answer.W"][:, pred]
    print("I answer %s (score %.3f)." % (CLASSES[pred] if pred < len(CLASSES) else pred, logits[pred]))
    print("Why, by named concept (fire x weight = contribution):")
    for k in np.argsort(-contrib)[:top_k]:
        c = cfg["middle_layer"][k]
        print("  %-24s %.2f x %+.3f = %+.3f" % (c["name"], h[k], t["answer.W"][k, pred], contrib[k]))
        print("      formula: %s" % c["formula"])
    for k in np.argsort(contrib)[:2]:
        if contrib[k] < 0:
            print("  AGAINST: %-20s %.2f x %+.3f = %+.3f" % (cfg["middle_layer"][k]["name"], h[k], t["answer.W"][k, pred], contrib[k]))
    print("bias %+.3f + all contributions %+.6f = %.6f | score %.6f | residual %.2e" % (
        t["answer.b"][pred], contrib.sum(), t["answer.b"][pred] + contrib.sum(), logits[pred],
        abs(t["answer.b"][pred] + contrib.sum() - logits[pred])))


if __name__ == "__main__":
    t, cfg = load()
    rng = np.random.default_rng(0)
    # a synthetic "garment": brighter top half, a vertical band in the middle
    img = np.zeros((28, 28), dtype=np.float32)
    img[3:16, 8:20] = 0.85
    img[16:25, 11:17] = 0.55
    img += rng.standard_normal((28, 28)).astype(np.float32) * 0.02
    explain(t, cfg, img.ravel())
