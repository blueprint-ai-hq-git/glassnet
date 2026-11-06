# DATA_LICENSES.md — every dataset used by GlassNet

Project rule (Yahya, 2026-10-01): ONLY these licenses may enter the project —
**MIT, Apache-2.0, CC0, CC BY 4.0**. Anything with an unclear license is
BLOCKED and listed here with the reason, until Yahya explicitly approves it.

---

## Tiny Shakespeare (Rung T2 — tiny next-word-prediction) — USED ✅

- **What:** 202,651 words / 40,000 lines / 1,115,394 bytes of Shakespeare
  plays (the classic tiny-LM corpus used by karpathy's char-rnn and
  nanoGPT). File: `data/tiny_shakespeare/input.txt`.
- **Content license = PUBLIC DOMAIN.** William Shakespeare died 1616; the
  text predates copyright in every jurisdiction and no rights holder
  exists. Public domain is the strictest possible "safe" state (the CC0
  ideal, which only mimics it).
- **Distribution repo = MIT.** The file is hosted in
  https://github.com/karpathy/char-rnn; the repo's own `Readme.md` states
  "## License — MIT" (line 139-141 of the master readme, fetched and
  verified 2026-10-02 by direct get of
  raw.githubusercontent.com/karpathy/char-rnn/master/Readme.md). The repo
  has no separate LICENSE file but its README is the explicit grant.
- **Decision record:** PD content is not literally on the 4-license list;
  Yahya approved use of tiny_shakespeare on 2026-10-02 via an explicit
  choice ("tiny_shakespeare — PD content, verified, go now") after seeing
  this exact verification report. sha256 of the downloaded file:
  `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
  (size 1,115,394 B matches the repo listing exactly).
- **Note:** char-level stats were the original use; we use it WORD-level
  (windowed next-word prediction). Splits: contiguous positional
  train 90% / val 5% / test 5% of the token stream (no shuffling —
  temporal LM convention), vocabulary built from TRAIN tokens only.

---

## Fashion-MNIST — USED ✅ (Phase 1 baseline and everything after)

- **What:** 60,000 training / 10,000 test images, 28x28 grayscale, 10 clothing
  classes (t-shirt/top, trouser, pullover, dress, coat, sandal, shirt,
  sneaker, bag, ankle boot).
- **Source:** https://github.com/zalandoresearch/fashion-mnist
  Files pulled from `data/fashion/` in that repo.
- **License: MIT** — Copyright (c) 2017 Zalando SE. Verified 2026-10-01 by
  reading the repo's LICENSE file (full text below).
- **Paper:** Han Xiao, Kashif Rasul, Roland Vollgraf, "Fashion-MNIST: a Novel
  Image Dataset for Benchmarking Machine Learning Algorithms", arXiv:1708.07747.

```
The MIT License (MIT)
Copyright © 2017 Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Note: `data/fashion_mnist_cache.npz` is a derived copy (pixel values
re-normalised to [0,1]) of the same MIT-licensed data.

---

## MNIST — BLOCKED ⛔ (NOT used)

- **Source:** https://yann.lecun.com/exdb/mnist/
- **Status checked 2026-10-01:** the official page contains **no license
  statement at all**. The dataset is widely redistributed for research, but
  the project rule says *never use data with unclear licenses*.
- **Action:** excluded from the project until a clear license is confirmed.
  The Phase 1 gate only requires Fashion-MNIST, so nothing is blocked by this.

---

## CIFAR-10 / CIFAR-100 — BLOCKED ⛔ (NOT used)

- **Source:** https://www.cs.toronto.edu/~kriz/cifar.html (official page)
- **Status checked 2026-10-01 (autonomous-loop Rung 2 license check):** the
  official page contains **no license statement at all** — only "please cite
  the tech report". Not on the allowed list (MIT, Apache 2.0, CC0,
  CC BY 4.0); per the safety rules the dataset was skipped and **nothing was
  downloaded**.
- **Action:** excluded. Rung 2 uses MedMNIST/PathMNIST instead.

---

## Rung 3 — Banking77 (tiny text task) — USED ✅

- **What:** 13,083 banking-customer short utterances, 77 intent classes
  ("card not working", "transfer fee", "top up by card", ...). The tiny
  text task for rung 3.
- **Source:** https://huggingface.co/datasets/PolyAI/banking77 (authors:
  PolyAI-LDN / Casanueva et al., "Efficient Intent Classification with
  Sparse User Intent Networks", arXiv:2108.02608, 2020).
- **License: CC BY 4.0** — verified 2026-10-01 via the HuggingFace dataset
  registry API: `https://huggingface.co/api/datasets/PolyAI/banking77` →
  `cardData.license = cc-by-4.0`. On the allowed list.
- Vocabulary and concept normalisation are computed on the TRAIN portion
  only (validation/test never leak into token selection or scaling).
- Derived cache: `data/banking77/banking77.csv` (+ loop caches).

- **What:** 18 standardized biomedical image datasets (2D/3D), 28×28,
  with official train/validation/test splits. We use **PathMNIST**: 9-class
  colorectal-cancer tissue classification — 89,996 train / 10,000 val /
  7,180 test.
- **Source:** https://zenodo.org/records/6496656 (the only official
  distribution link, per medmnist.com). File used:
  `pathmnist.npz` (direct Zenodo file URL).
- **License: CC BY 4.0** — verified 2026-10-01 directly from the official
  Zenodo record API (`/api/records/6496656` → metadata.license.id =
  `cc-by-4.0`). Allowed by the project rule (MIT, Apache 2.0, CC0,
  CC BY 4.0).
- **Publication:** Yang et al., "MedMNIST v2: A Large-Scale Lightweight
  Benchmark for 2D and 3D Biomedical Image Classification", Scientific Data
  10:41 (2023).
- Class labels (PathMNIST): 0 adipose, 1 background, 2 debris,
  3 lymphocytes, 4 mucus, 5 smooth muscle, 6 normal colon mucosa,
  7 cancer-associated stroma, 8 colorectal adenocarcinoma epithelium.
- Preprocessing (documented, applied by our loader): RGB → greyscale by
  channel mean, /255 → [0,1], flattened to 784 — the same pixel space as
  Fashion-MNIST, so the documented concept formulas apply unchanged.
- Derived cache: `data/loop_pathmnist_gray.npz` (greyscale+normalised copy).
## Chat datasets (fetched 2026-10-07, for the conversational model)

| dataset | license | allowed by the project rule | used for |
|---|---|---|---|
| OpenAssistant/oasst1 | Apache 2.0 | YES | real human assistant conversations |
| Anthropic/hh-rlhf | MIT | YES | human helpful/harmless dialogues |
| HuggingFaceH4/ultrachat_200k | MIT | YES | instruction/response pairs |

All three are on the owner's allowed list (MIT, Apache 2.0, CC0, CC BY 4.0). None is scraped, none
is behind a paid API, and nothing is sent anywhere -- these are direct downloads of published,
licensed datasets.

Previously used and still on record: tiny_shakespeare (public domain) for the concept-layer work,
Fashion-MNIST (MIT) for the earlier image work.

## Code datasets (fetched 2026-10-07, for the code half of the goal)

| dataset | license | allowed | used for |
|---|---|---|---|
| ise-uiuc/Magicoder-OSS-Instruct-75K | MIT | YES | instruction -> code pairs |
| codeparrot/apps | MIT | YES | problem -> solution, with test cases |
| codeparrot/github-code-clean | Apache 2.0 | YES (noted) | raw source files, 880 parquet shards |

Checked BEFORE download and recorded:
- **SKIPPED, licence unclear or absent:** codeparrot/codeparrot-clean (no licence stated),
  bigcode/the-stack-smol (no licence stated), bigcode/starcoderdata (licence "other"),
  HuggingFaceH4/CodeAlpaca_20K (licence "cc" -- not one of the allowed four).
- **SKIPPED, gated:** bigcode/the-stack-dedup, nampdn-ai/tiny-codes.
- **Unauthorized (401):** microsoft/CodeXGLUE, google-deepmind/mbpp, TIGER-Lab/CodeFeedback.
- **Allowed and available but not yet used:** openai/openai_humaneval (MIT),
  deepmind/code_contests (CC BY 4.0), open-r1/codeforces (CC BY 4.0).

## Checked 2026-10-08 — COCO (image captioning candidate) — **SKIPPED**
Official licence terms: the ANNOTATIONS are CC BY 4.0, but the CONSORTIUM DOES NOT OWN THE IMAGE
COPYRIGHTS — the ~330K photos stay under the original Flickr terms of use, and the user "accepts
full responsibility for the use of the dataset including any copies made from it." A third-party
web claim that "the images are CC BY 4.0" contradicts the official statement and was NOT relied on.
Under the project rules (allowed set MIT/Apache-2.0/CC0/CC-BY-4.0; skip unclear; commercial
provenance required) the IMAGE side is exactly the unclear class. NOT DOWNLOADED.
Same status as all checked face datasets (UTKFace/LFW/CelebA/FFHQ: unclear or non-commercial).
**Open-vocabulary captioning therefore awaits a licence-clean source.** Options on the table:
Conceptual Captions 3.0 (Apache-2.0 annotations but same URL-image caveat), CC-STORE /
SA-1B-type sets (non-commercial research terms — fails the rule), or owner-supplied/approved
photos. This is a data-sourcing decision, not a model limitation, and it is reported as such.

---

## EXTERNAL DATA WINDOW (2026-10-14, Wi-Fi available) — license gate run BEFORE download

Method: for every candidate, the Hugging Face dataset API (`https://huggingface.co/api/datasets/<id>`)
was queried and the repo's DECLARED license read. Only `mit`, `apache-2.0`, `cc0-1.0`, `cc-by-4.0`
were downloaded. Nothing with an undeclared or other license was fetched.

| dataset | declared license | role | verdict |
|---|---|---|---|
| openai/gsm8k | MIT | grade-school math benchmark | DOWNLOADED |
| cais/mmlu | MIT | multi-subject knowledge benchmark | DOWNLOADED |
| openai/openai_humaneval | MIT | code benchmark | DOWNLOADED |
| google-research-datasets/mbpp | CC-BY-4.0 | code benchmark | DOWNLOADED |
| OpenAssistant/oasst1 | Apache-2.0 | 160k real human chat messages (training data) | DOWNLOADED |
| HuggingFaceH4/ultrachat_200k | MIT | large multi-turn chat corpus (training data) | DOWNLOADED |
| microsoft/orca-math-word-problems-200k | MIT | math word problems (training data) | DOWNLOADED |
| ylecun/mnist | MIT | digits (image line) | DOWNLOADED |
| google/speech_commands | CC-BY-4.0 | real spoken audio (audio line) | DOWNLOADED |
| Rowan/hellaswag | NOT DECLARED | commonsense benchmark | SKIPPED (rule 3) |
| allenai/winogrande | NOT DECLARED | coreference benchmark | SKIPPED (rule 3) |
| roneneldan/TinyStories | CDLA-Sharing-1.0 | simple fluent English | SKIPPED (not on the allowed list) |
| uoft-cs/cifar10 | unknown | natural images | SKIPPED (rule 3) |
| medmnist/bloodmnist | (API 401) | medical volume | SKIPPED (not verifiable this way) |

**MNIST note (this resolves the earlier block):** the original MNIST webpage states no license,
which is why the line was previously BLOCKED. The author's own mirror `ylecun/mnist` on Hugging
Face declares **MIT**. That is a grant from the dataset's original author, so MNIST is now
ADMISSIBLE, with the provenance recorded exactly as: original page license = none stated;
author's HF mirror declared license = MIT (recorded 2026-10-14); requirements "match the training
set distribution".

**Retrieval:** all downloads go through the `hf` CLI (snapshot download) into
`data/external/<name>/`, inside the project folder on Hard Disk 4. No site scraping, no paid APIs,
no data sent anywhere. Sizes and checksums are appended after the transfer completes.

### Download sizes and the base-corpus license (recorded 2026-10-14, after transfer)

| dataset | size on disk | license (declared) |
|---|---|---|
| Anthropic/hh-rlhf (the talk base corpus source, already local) | 29 MB (2 gz files) | **MIT** |
| cais/mmlu | 259 MB | MIT |
| HuggingFaceH4/ultrachat_200k | 1.6 GB | MIT |
| OpenAssistant/oasst1 | 221 MB | Apache-2.0 |
| microsoft/orca-math-word-problems-200k | 81 MB | MIT |
| ylecun/mnist | 18 MB | MIT |
| openai/gsm8k | 5.7 MB | MIT |
| openai/openai_humaneval | 104 KB | MIT |
| google-research-datasets/mbpp | 400 KB | CC-BY-4.0 |
| google/speech_commands | 2.26 GB (Google archive; HF repo carries only a loading script) | CC BY 4.0 |

Total external: ~4.5 GB (project disk budget for growth: 20 GB; 153 GB free).
Note: `google/speech_commands` on HF holds only README + `.gitattributes` + a loading script
(verified over the HTTP API: 3 files), so the audio data was taken from Google's own published
archive instead, which is the dataset's canonical CC BY 4.0 source.

### Merged talk corpus built from the licensed sources (2026-10-14)
    data/chat/prepared_ext/: 156,521 conversations / 38,829,582 tokens  (2.9x the previous corpus)
    tokens per source: hh 13,318,447 | oasst1 7,511,638 | ultrachat 11,999,719 | orca_math 5,999,778
    vocabulary: rebuilt from TRAIN text only (12,000 words) -- same rule as every other corpus
    leak fix AT BUILD TIME: 867 val conversations whose 100-token window appears in train were
    moved to train (applying the splitfix finding from the start); val = 2,262 conversations

### Additional licensed downloads (2026-10-14, second pass — bulk gate over 27 candidates)

| dataset | license | size | role |
|---|---|---|---|
| Open-Orca/OpenOrca | MIT | 3.9 GB | instruction mix (training) |
| Open-Orca/SlimOrca | MIT | 0.94 GB (JSONL/sharegpt) | curated instruction (training) |
| meta-math/MetaMathQA | MIT | 0.38 GB (plain JSON) | math reasoning (training) |
| nvidia/OpenMathInstruct-2 | CC-BY-4.0 | 6.7 GB (partial: 25 of 32 shards) | math reasoning (training) |
| OpenAssistant/oasst2 | Apache-2.0 | 0.33 GB | real human chat (training) |
| truthful_qa | Apache-2.0 | small | truthfulness benchmark |
| google/speech_commands | CC BY 4.0 | 5.5 GB (archive + extracted) | 105,835 real speech clips (audio line) |

SKIPPED at the gate (license): lukaemon/bbh (none declared), allenai/openbookqa (unknown),
allenai/ai2_arc (CC BY-SA 4.0), google/boolq (CC BY-SA 3.0), winogrande (none), Rowan/hellaswag
(none). SKIPPED as too large for this window (reason recorded): cerebras/SlimPajama-627B,
allenai/OLMo-mix-1124, HuggingFaceTB/finemath.
Disk note: `data/external` reached the 20 GB self-imposed guard, so the tail of the bulk download
was STOPPED rather than quietly exceeded. Yahya then said the Wi-Fi is unlimited and to download
everything, so the guard was raised to 30 GB (stated in lens/scripts/bulk_license_and_download.py)
and the download resumed. Disk free at that point: 133 GB.

## Knowledge / maths additions - 2026-10-18

Licence gate was run against Hugging Face's own metadata for every candidate, and the answer is recorded
here whether it was allowed or skipped. Nothing with an unclear licence is used, and nothing gated is
forced.

| dataset | licence | decision | why |
|---|---|---|---|
| meta-math/MetaMathQA | MIT | ALLOWED | maths word problems with worked solutions |
| EleutherAI/hendrycks_math (algebra) | MIT | ALLOWED | competition maths, worked solutions |
| open-r1/OpenR1-Math-220k | Apache-2.0 | ALLOWED | maths, one shard |
| HuggingFaceTB/cosmopedia (khanacademy, openstax, stanford, auto_math_text) | Apache-2.0 | ALLOWED | educational/textbook-like; public; carries real prompts |
| sedthh/gutenberg_english | MIT | ALLOWED | public-domain prose |
| nampdn-ai/tiny-textbooks | Apache-2.0 | **SKIPPED** | licence is fine but the dataset is GATED: HF answered "Access to dataset nampdn-ai/tiny-textbooks is restricted. You must have access to it and be authenticated to access it." Substituted with cosmopedia above rather than using an account to force it |
| HuggingFaceTB/finemath | ODC-By | SKIPPED | licence not in the allowed list |
| allenai/peS2o | ODC-By | SKIPPED | licence not in the allowed list |
| togethercomputer/RedPajama-Data-1T | NOT STATED | SKIPPED | no licence stated |
| Wikipedia | CC BY-SA + GFDL | SKIPPED | share-alike terms not in the allowed list |

Never train on: GSM8K, MMLU, HumanEval, MBPP (our exam papers).

Synthetic-prompt note: cosmopedia subsets carry genuine source prompts, so their human side is real.
Gutenberg book text has no questions in it, so its prompt side is generated from the passage's own
opening words and is labelled synthetic in the corpus meta file.
