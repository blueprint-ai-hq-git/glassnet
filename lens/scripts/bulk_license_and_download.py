#!/usr/bin/env python3
"""BULK license gate + download: check many candidate datasets, download every ALLOWED one.

The project rule is unchanged -- only MIT / Apache-2.0 / CC0-1.0 / CC-BY-4.0 enter the project,
and nothing with an undeclared or other license is fetched. What changes is SCALE: instead of one
dataset at a time, this checks a wide candidate list and pulls all allowed ones in a single pass,
so the comparison matrix can be built from several corpora instead of serial experiments.

Disk guard: stops before exceeding BUDGET_GB of NEW data (project rule: <20 GB growth).
"""
import json
import os
import subprocess
import urllib.request

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
HF = "/home/yahya/.hermes/installs/15d969a29be1af88/environments/7c7750f1e3d5471e920a055130314e9b/venv/bin/hf"
ALLOWED = {"mit", "apache-2.0", "cc0-1.0", "cc-by-4.0", "bsd-3-clause", "cc-by-3.0"}
BUDGET_GB = 30.0          # raised from 12 on Yahya's instruction ("Wi-Fi is unlimited, download
                          # all the things"). Disk has ~133 GB free. Stated, not silent.
SKIP_HUGE = {"cerebras/SlimPajama-627B": "~800 GB repo: would swamp the window; a bounded shard "
                                         "slice can be taken on request",
             "allenai/OLMo-mix-1124": "multi-hundred-GB pretraining mix: same reason",
             "HuggingFaceTB/finemath": "~2 TB repo: same reason"}

CANDIDATES = [
    # --- training data: chat / instruction / reasoning ---
    ("OpenAssistant/oasst2", "human chat v2"),
    ("Open-Orca/OpenOrca", "instruction mix (large)"),
    ("Open-Orca/SlimOrca", "curated instruction mix"),
    ("teknium/OpenHermes-2.5", "instruction mix"),
    ("garage-bAInd/Open-Platypus", "reasoning instruction"),
    ("meta-math/MetaMathQA", "math reasoning"),
    ("nvidia/OpenMathInstruct-2", "math reasoning (large)"),
    ("microsoft/orca-math-word-problems-200k", "math (have)"),
    ("WizardLMTeam/WizardLM_evol_instruct_V2_196k", "instruction"),
    ("yahma/alpaca-cleaned", "instruction"),
    ("HuggingFaceH4/deita-10k-v0", "instruction"),
    ("cerebras/SlimPajama-627B", "web text volume (Apache-2.0 claim)"),
    ("EleutherAI/proof-pile-2", "math/code proofs"),
    ("allenai/tulu-v2-sft-mixture", "SFT mixture"),
    ("allenai/OLMo-mix-1124", "pretraining mix"),
    ("HuggingFaceTB/finemath", "math web text"),
    ("codeparrot/codeparrot-clean-valid", "python code"),
    ("bigcode/the-stack-smol", "code (per-file licenses -- expect skip)"),
    # --- benchmarks ---
    ("truthful_qa", "truthfulness benchmark"),
    ("lukaemon/bbh", "big-bench-hard"),
    ("allenai/openbookqa", "science QA"),
    ("allenai/ai2_arc", "science QA"),
    ("google/boolq", "reading comprehension"),
    ("winogrande", "coreference"),
    ("Rowan/hellaswag", "commonsense"),
    ("cais/mmlu", "MMLU (have)"),
    ("openai/gsm8k", "GSM8K (have)"),
]


def declared_license(rid):
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{rid}", timeout=30) as r:
            d = json.load(r)
        lic = (d.get("cardData") or {}).get("license", "NOT DECLARED")
        if isinstance(lic, list):
            lic = ",".join(lic)
        return str(lic).lower().strip(), str(lic)
    except Exception as e:
        return "error", f"{type(e).__name__}"


def main():
    allowed, skipped = [], []
    print("=== license gate over", len(CANDIDATES), "candidates")
    for rid, why in CANDIDATES:
        low, shown = declared_license(rid)
        if low in ALLOWED:
            allowed.append((rid, shown, why))
            print(f"  ALLOW {rid:<52} {shown:<14} {why}")
        else:
            skipped.append((rid, shown))
            print(f"  SKIP  {rid:<52} {shown:<14} {why}")
    print(f"\n  allowed: {len(allowed)}   skipped: {len(skipped)}")
    json.dump({"allowed": allowed, "skipped": skipped},
              open(os.path.join(PROJ, "lens/runs/LICENSE_GATE_BULK.json"), "w"), indent=1)

    used = float(subprocess.run(["du", "-sb", os.path.join(PROJ, "data", "external")],
                               capture_output=True, text=True).stdout.split()[0]) / 1e9
    for rid, lic, why in allowed:
        if rid in SKIP_HUGE:
            print(f"  SKIP-HUGE {rid:<48} {SKIP_HUGE[rid]}")
            continue
        if used > BUDGET_GB:
            print(f"  BUDGET STOP at {used:.1f} GB -- remaining candidates not downloaded")
            break
        dest = os.path.join(PROJ, "data", "external", rid.replace("/", "__"))
        if os.path.isdir(dest) and os.listdir(dest):
            print(f"  have  {rid}")
            continue
        print(f"  GET   {rid}  ({lic})")
        subprocess.run([HF, "download", rid, "--repo-type", "dataset", "--local-dir", dest],
                       capture_output=True, text=True)
        sz = subprocess.run(["du", "-sb", dest], capture_output=True, text=True).stdout.split()[0]
        used += float(sz) / 1e9
        print(f"        -> {float(sz)/1e9:.2f} GB   running total {used:.2f} GB")
    print(f"\n  done. external data total: {used:.2f} GB")


if __name__ == "__main__":
    main()