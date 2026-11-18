
## Both published artifacts now verify from the Hub bytes alone (2026-11-18)

The canonical-hash fix was applied to the second published model, and then both were verified **from the downloaded bytes** rather than from the upload calls.

```
glassnet-scan-r9-recurrent   (chat / recurrent)
  remote files: .gitattributes, LICENSE, MANIFEST.json, README.md, config.json, model.safetensors
  published canonical sha256  6dce6567eefa0c6b386fa97101834fb86aa5ebef26631dfd98f34ceeab9aacc4
  recomputed from download    6dce6567eefa0c6b386fa97101834fb86aa5ebef26631dfd98f34ceeab9aacc4   MATCH
  held-out accuracy recomputed from the DOWNLOADED weights: 16.27% over 756 scored positions
  published figure: 16.27%   <- reproduces EXACTLY, from the artifact alone

glassnet-readable-vision-21   (readable image model)
  canonical sha256 matches, and the model rebuilt from the downloaded bytes in plain numpy scores 85.50% over 2,000 test images
```

The chat model's number reproducing **to the digit** from the published file is the strongest check available here: the published figure is not a claim about a checkpoint on this
machine, it is a property of the artifact anyone can download.

`tools/verify_hf_scan.py` and `tools/verify_hf_vision.py` are both in the tree so this can be re-run at any time, and each one fails loudly if the canonical hash disagrees.

### Why this matters beyond bookkeeping

Two of the project's numbers already had to be corrected on the page (a parameter count, and a gate metric that turned out to be blind). Publishing a model that cannot be checked
remotely puts every claim it carries on the same footing as those corrected ones. Now the claim and the check travel together.
