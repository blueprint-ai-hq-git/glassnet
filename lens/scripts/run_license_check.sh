#!/bin/bash
# LICENSE + SIZE CHECK for every dataset candidate. Nothing is downloaded until its declared
# license is MIT / Apache-2.0 / CC0 / CC-BY-4.0 (the project's rule). Unclear or other licenses
# are printed as SKIP, never downloaded. This is the gate the project requires, run BEFORE bytes.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import json, urllib.request
CAND = [
    ("openai/gsm8k", "math word problems (grade-school)"),
    ("cais/mmlu", "multi-subject knowledge benchmark"),
    ("openai/openai_humaneval", "code benchmark"),
    ("google-research-datasets/mbpp", "code benchmark"),
    ("Rowan/hellaswag", "commonsense benchmark"),
    ("allenai/winogrande", "coreference benchmark"),
    ("OpenAssistant/oasst1", "real human chat (train data)"),
    ("HuggingFaceH4/ultrachat_200k", "large chat corpus (train data)"),
    ("roneneldan/TinyStories", "simple fluent English stories (train data)"),
    ("microsoft/orca-math-word-problems-200k", "math volume (train data)"),
    ("google/speech_commands", "spoken commands (audio train data)"),
    ("openslr/librispeech_asr", "read speech (audio train data)"),
    ("medmnist/bloodmnist", "blood cell images (medical volume)"),
    ("uoft-cs/cifar10", "natural images (image line volume)"),
    ("ylecun/mnist", "digits (image line)"),
]
ALLOWED = {"mit", "apache-2.0", "cc0-1.0", "cc-by-4.0", "cc-by-4.0"}
for rid, why in CAND:
    url = f"https://huggingface.co/api/datasets/{rid}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
        lic = (d.get("cardData") or {}).get("license", "NOT DECLARED")
        if isinstance(lic, list):
            lic = ",".join(lic)
        low = str(lic).lower().strip()
        verdict = "ALLOW" if low in ALLOWED else "SKIP "
        print(f"  {verdict} {rid:<46} license: {lic:<14} -- {why}")
    except Exception as e:
        print(f"  ERROR {rid:<46} {type(e).__name__}: {e}")
PY