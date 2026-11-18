# The safetensors file hash is not stable across rebuilds - verify the canonical hash instead (2026-11-18)

Publishing the readable vision model kept producing a **different `sha256` for the `.safetensors` file on every rebuild**, with the same checkpoint, the same tensors and the same
metadata. Three pushes gave `c83f8410...`, `d6955049...`, `d853b88e...`. A manifest whose hash changes when nothing changed is not an integrity check, it is decoration, so this was
tracked down.

## What is actually unstable

Three builds from the same arrays, same process:

```
build 0: header fdbecdd9703d613a (512 bytes) | DATA 853dfd0ad0b6999b (63908 bytes) | canonical 9ea18a1b41897ce9
build 1: header fdbecdd9703d613a (512 bytes) | DATA 853dfd0ad0b6999b (63908 bytes) | canonical 9ea18a1b41897ce9
build 2: header 112ac390d0b4880b (512 bytes) | DATA 853dfd0ad0b6999b (63908 bytes) | canonical 9ea18a1b41897ce9
```

**The tensor data is byte-identical every time** (`853dfd0a...`, 63,908 bytes) and a canonical hash over sorted `name + dtype + shape + bytes` is stable (`9ea18a1b...`). Only the
**512-byte JSON header** differs between builds. Metadata key ordering was tested and is **not** the cause - writing the same metadata in a different order produced the identical
file. So the header itself is not byte-stable in this `safetensors` version.

## The consequence, and the fix

Every published artifact now carries **two** hashes, labelled:

- **weights sha256 (canonical)** - sorted name + dtype + shape + bytes. **Stable across rebuilds; this is the value to verify.**
- **container file sha256** - this exact file. Documented as explicitly **rebuild-unstable**, so nobody treats a mismatch as corruption.

## Verified from the Hub, not from the upload call

`tools/verify_hf_vision.py` downloads the published model and checks it end to end:

```
remote files: .gitattributes, LICENSE, MANIFEST.json, README.md, config.json, explain_image.py, model.safetensors
published canonical sha256:    9ea18a1b41897ce9b1cb570117ae3387a4e1e50e361029bc5817dfa4b35a6d9d
recomputed from the download:  9ea18a1b41897ce9b1cb570117ae3387a4e1e50e361029bc5817dfa4b35a6d9d
CANONICAL MATCH: YES
held-out accuracy recomputed from the DOWNLOADED weights: 85.50% over 2000 images
```

That last line is the important one: the model is rebuilt **from the downloaded bytes only** - plain numpy, the published `config.json` for shapes, no local training code - and it
scores 85.50% on the first 2,000 test images (against 84.62% on a different random 4,000-image sample recorded earlier; both are honest samples of the same model).

## Also shipped with the artifact

`explain_image.py` now travels inside the model repo, so downloading the model is enough to ask it *why*: it prints the answer, the named concepts that produced it, each concept's
formula, the concepts that argued against it, and the arithmetic showing the contributions equal the score exactly. Verified running from the published repo contents with
residual **0.00e+00**.
