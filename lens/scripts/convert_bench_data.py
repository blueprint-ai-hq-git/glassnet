#!/usr/bin/env python3
"""Convert the downloaded benchmark parquet into plain JSONL, ONCE, with the data venv.

Why: pyarrow lives in .venv-data (data I/O only) and must NOT be installed into .venv-gpu, where
the validated numpy/CuPy stack backs every training and audit run. Converting once keeps the two
environments separate and makes the eval itself dependency-free.
"""
import json
import os

import pyarrow.parquet as pq

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
EXT = os.path.join(PROJ, "data", "external")


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {path} ({len(rows):,} rows)")


def mmlu():
    rows = []
    base = os.path.join(EXT, "mmlu")
    for root, _, fs in os.walk(base):
        for fn in sorted(fs):
            if fn.endswith(".parquet") and "test" in fn:
                t = pq.read_table(os.path.join(root, fn),
                                  columns=["question", "choices", "answer", "subject"])
                for i in range(t.num_rows):
                    rows.append({"question": t["question"][i].as_py(),
                                 "choices": t["choices"][i].as_py(),
                                 "answer": int(t["answer"][i].as_py()),
                                 "subject": t["subject"][i].as_py()})
    write_jsonl(os.path.join(EXT, "mmlu_test.jsonl"), rows)


def gsm8k():
    t = pq.read_table(os.path.join(EXT, "gsm8k", "main", "test-00000-of-00001.parquet"),
                      columns=["question", "answer"])
    write_jsonl(os.path.join(EXT, "gsm8k_test.jsonl"),
                [{"question": t["question"][i].as_py(), "answer": t["answer"][i].as_py()}
                 for i in range(t.num_rows)])


def code():
    rows = []
    base = os.path.join(EXT, "humaneval")
    for root, _, fs in os.walk(base):
        for fn in fs:
            if fn.endswith(".parquet"):
                t = pq.read_table(os.path.join(root, fn))
                cols = t.column_names
                pk = "prompt" if "prompt" in cols else cols[0]
                for i in range(t.num_rows):
                    rows.append({"source": "humaneval", "task": t[pk][i].as_py()})
    base = os.path.join(EXT, "mbpp")
    for root, _, fs in os.walk(base):
        for fn in fs:
            if fn.endswith(".parquet"):
                t = pq.read_table(os.path.join(root, fn))
                cols = t.column_names
                tk = "text" if "text" in cols else cols[0]
                for i in range(t.num_rows):
                    rows.append({"source": "mbpp", "task": t[tk][i].as_py()})
    write_jsonl(os.path.join(EXT, "code_bench.jsonl"), rows)


if __name__ == "__main__":
    mmlu()
    gsm8k()
    code()