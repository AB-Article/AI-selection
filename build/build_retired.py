#!/usr/bin/env python3
"""
build_retired.py -- assemble the retired-benchmark ladder corpus.

Source: EleutherAI/pythia, evals/ directory (public GitHub repository).  It
holds lm-evaluation-harness runs for six fully released model suites -- Pythia
v1, Pythia v1 deduped, Pythia v0, Pythia v0 deduped, OPT and BLOOM -- on eight
accuracy-scored benchmarks, under one harness, one prompt format and one shot
count.  For the Pythia suites the same eight benchmarks are also scored at
intermediate training checkpoints, which gives a second resource axis.

Writes  corpus_retired.csv   one row per (suite, size, step, shots, task)
"""
import csv
import glob
import json
import os
import re
import sys
import tarfile
import urllib.request

OUT = os.environ.get("OUT", "results")
SRC = os.environ.get("PYTHIA_SRC", "")
URL = "https://codeload.github.com/EleutherAI/pythia/tar.gz/refs/heads/main"

# accuracy-scored tasks and their declared chance levels
TASKS = {
    "sciq":           ("SciQ",          0.25),
    "piqa":           ("PIQA",          0.50),
    "arc_easy":       ("ARC-Easy",      0.25),
    "lambada_openai": ("LAMBADA",       0.00),
    "winogrande":     ("Winogrande",    0.50),
    "arc_challenge":  ("ARC-Challenge", 0.25),
    "logiqa":         ("LogiQA",        0.25),
    "wsc":            ("WSC",           0.50),
}

# total parameters, as released
PARAMS = {
    "pythia-70m": 70426624, "pythia-160m": 162322944, "pythia-410m": 405334016,
    "pythia-1b": 1011781632, "pythia-1.4b": 1414647808,
    "pythia-2.8b": 2775208960, "pythia-6.9b": 6857302016,
    "pythia-12b": 11846072320,
    "opt-125m": 125e6, "opt-350m": 350e6, "opt-1.3b": 1.3e9, "opt-2.7b": 2.7e9,
    "opt-6.7b": 6.7e9, "opt-13b": 13e9, "opt-30b": 30e9, "opt-66b": 66e9,
    "bloom-560m": 559e6, "bloom-1b1": 1065e6, "bloom-1b7": 1722e6,
    "bloom-3b": 3003e6, "bloom-7b1": 7069e6,
}
# non-embedding parameters, Pythia paper Table 1 (robustness axis)
NONEMB = {
    "pythia-70m": 18915328, "pythia-160m": 85056000, "pythia-410m": 302311424,
    "pythia-1b": 805736448, "pythia-1.4b": 1208602624,
    "pythia-2.8b": 2517652480, "pythia-6.9b": 6444163072,
    "pythia-12b": 11327027200,
}
TOK_PER_STEP = 1024 * 2048          # Pythia batch: 1024 sequences of 2048
FINAL_STEP = 143000


def fetch():
    if SRC and os.path.isdir(SRC):
        return SRC
    local = os.path.join(OUT, "_pythia_src")
    if os.path.isdir(os.path.join(local, "pythia-main", "evals")):
        return os.path.join(local, "pythia-main")
    os.makedirs(local, exist_ok=True)
    tgz = os.path.join(local, "pythia.tar.gz")
    if not os.path.exists(tgz):
        print("downloading EleutherAI/pythia ...", file=sys.stderr)
        urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(local)
    return os.path.join(local, "pythia-main")


def acc(block):
    """Harness accuracy, preferring plain acc; lambada reports acc only."""
    for key in ("acc",):
        if key in block:
            return float(block[key])
    return None


def emit(rows, suite, size, step, shots, path):
    try:
        res = json.load(open(path))["results"]
    except Exception:
        return
    key = re.sub(r"^(pythia|opt|bloom)-", "", size)
    for stem, (label, chance) in TASKS.items():
        if stem not in res:
            continue
        a = acc(res[stem])
        if a is None or not (0.0 < a < 1.0):
            continue
        rows.append(dict(
            suite=suite, model=size, task=label, harness_task=stem,
            chance=chance, shots=shots, step=step,
            params=PARAMS.get(size, ""), nonemb=NONEMB.get(size, ""),
            train_tokens=(step * TOK_PER_STEP) if step else "",
            acc=round(a, 6), key=key))


def main():
    root = fetch()
    ev = os.path.join(root, "evals")
    rows = []

    # ---- Pythia v1 and v0, capacity ladder (final step) and token ladders --
    for version, tag in (("pythia-v1", "v1"), ("pythia-v0", "v0")):
        base = os.path.join(ev, version)
        if not os.path.isdir(base):
            continue
        for d in sorted(os.listdir(base)):
            m = re.fullmatch(r"pythia-(\d+\.?\d*[mb])(-deduped)?", d)
            if not m:                       # skip 1MtokBS, bf16, winobias ...
                continue
            size = f"pythia-{m.group(1)}"
            suite = f"Pythia {tag}{' deduped' if m.group(2) else ''}"
            for shots, sub in ((0, "zero-shot"), (5, "five-shot")):
                for f in sorted(glob.glob(os.path.join(base, d, sub, "*.json"))):
                    s = re.search(r"step(\d+)", os.path.basename(f))
                    if not s:
                        continue
                    emit(rows, suite, size, int(s.group(1)), shots, f)

    # ---------------------------------- OPT and BLOOM, capacity ladder only --
    for sub, suite in (("opt", "OPT"), ("bloom", "BLOOM")):
        for f in sorted(glob.glob(os.path.join(ev, sub, "*.json"))):
            size = os.path.basename(f)[:-5]
            emit(rows, suite, size, FINAL_STEP if sub == "opt" else 0, 0, f)

    # Some v0 directories carry a five-shot copy whose harness output is
    # byte-identical to the zero-shot run; those are duplicates of the same
    # evaluation, not a second shot count, and are dropped.
    zero = {(r["suite"], r["model"], r["step"], r["task"]): r["acc"]
            for r in rows if r["shots"] == 0}
    keep, ndup = [], 0
    for r in rows:
        k = (r["suite"], r["model"], r["step"], r["task"])
        if r["shots"] and k in zero and abs(zero[k] - r["acc"]) < 1e-12:
            ndup += 1
            continue
        keep.append(r)
    rows = keep
    print(f"dropped {ndup} five-shot records identical to their zero-shot run",
          file=sys.stderr)

    cols = ["suite", "model", "key", "task", "harness_task", "chance", "shots",
            "step", "train_tokens", "params", "nonemb", "acc"]
    path = os.path.join(OUT, "corpus_retired.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda z: (z["suite"], z["task"],
                                             float(z["params"] or 0), z["shots"],
                                             z["step"])):
            w.writerow(r)
    print(f"wrote {path}: {len(rows)} rows, "
          f"{len({(r['suite'], r['model']) for r in rows})} checkpointed models, "
          f"{len({r['task'] for r in rows})} tasks")


if __name__ == "__main__":
    main()
