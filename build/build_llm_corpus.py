#!/usr/bin/env python3
"""
build_llm_corpus.py  --  assemble corpus_llm.csv, the task-resolved
language-model inference corpus.

The diagnosis in Section 12.3 of the manuscript was that the LLM frontiers
could not be identified because the quality axis was an *aggregate* of a dozen
benchmarks with different chance levels, which compressed the resolved
information of every front into about a quarter of a nat.  The remedy is to
keep the same measured latencies and replace the aggregate with the individual
task scores the aggregate is built from, each with its own declared chance
level, and to add two further measured sources.

Quality
-------
  Open LLM Leaderboard v1  ARC-Challenge (25-shot), HellaSwag (10-shot),
                           MMLU (5-shot), Winogrande (5-shot), GSM8K (5-shot).
                           Raw accuracies with the declared chance level of
                           each benchmark (1/4, 1/4, 1/4, 1/2, 0).
  Open LLM Leaderboard v2  IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-PRO.
                           The leaderboard publishes, besides the raw score, a
                           score normalised against the random baseline of the
                           benchmark; that normalised score is exactly 1 - eps
                           in our convention, so it is used with chance = 0.
  llama.cpp                WikiText perplexity, entered as eps = PPL/PPL_REF.
  Bench360                 MMLU accuracy, SQuAD-v2 F1, CNN/DailyMail ROUGE-L,
                           Spider execution accuracy, with declared chance.

Resource
--------
  llm-perf              end-to-end decode latency on four fixed platforms,
                        the same measurements already in corpus_extended.csv.
  llm-inference-bench   measured latency on nine accelerators under six
                        frameworks (argonne-lcf/LLM-Inference-Bench).
  llama.cpp             measured generation and prompt speed per quantisation.
  Bench360              time per output token under a 24 GB budget.

Perplexity convention
---------------------
The chance level for next-token prediction is a uniform choice among a
reference set of PPL_REF continuations, so eps = PPL/PPL_REF and
S = -ln eps = ln PPL_REF - ln PPL is the information resolved per token.
Because eps enters only through eps(t) = eps_inf + c t^-alpha, changing
PPL_REF rescales eps_inf and c and leaves alpha, hence lambda* = alpha/2,
exactly unchanged; stats_llm.py reports the sensitivity.
"""
import collections
import csv
import os
import re
import statistics
import sys

DEST = os.environ.get("DATA", "data")
D = os.environ.get("DATA", "data")
OUT = os.environ.get("OUT", "results")
PPL_REF = float(os.environ.get("PPL_REF", 100.0))

COLS = ["task", "dataset", "model", "device", "runtime", "precision", "cell",
        "metric", "Q", "chance", "latency_ms", "param_count", "gmacs", "source"]

# canonical serving configuration for the LLM-Inference-Bench cells; the other
# batch sizes and sequence lengths are used only by the sensitivity analysis
LIB_IO = os.environ.get("LIB_IO", "1024")
LIB_BATCH = os.environ.get("LIB_BATCH", "32")
LIB_ALL = os.environ.get("LIB_ALL", "0") == "1"

V1_TASKS = {
    "arc":        ("ARC-Challenge", 0.25),
    "hellaswag":  ("HellaSwag", 0.25),
    "mmlu":       ("MMLU", 0.25),
    "winogrande": ("Winogrande", 0.50),
    "gsm8k":      ("GSM8K", 0.00),
}
V2_TASKS = {
    "ifeval": "IFEval", "bbh": "BBH", "math": "MATH Lvl 5",
    "gpqa": "GPQA", "musr": "MuSR", "mmlu_pro": "MMLU-PRO",
}

rows = []


def load_scores():
    """model -> {(task label, chance): Q}, median over leaderboard rows."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))

    p1 = f"{DEST}/huggingface_v1.csv"
    if os.path.exists(p1):
        for r in csv.DictReader(open(p1)):
            m = r["model"].strip().lower()
            for col, (label, ch) in V1_TASKS.items():
                try:
                    v = float(r[col])
                except (TypeError, ValueError, KeyError):
                    continue
                if v > 0:
                    acc[m][(label, ch)].append(v / 100.0)

    p2 = f"{DEST}/huggingface_v2.csv"
    if os.path.exists(p2):
        for r in csv.DictReader(open(p2)):
            m = r["model_name"].strip().lower()
            for stem, label in V2_TASKS.items():
                try:
                    v = float(r[f"evaluations_{stem}_normalized_score"])
                except (TypeError, ValueError, KeyError):
                    continue
                if v > 0:
                    acc[m][(label, 0.0)].append(v / 100.0)

    return {m: {k: statistics.median(v) for k, v in d.items()}
            for m, d in acc.items()}


def source_llmperf(scores):
    n = 0
    for r in csv.DictReader(open(f"{D}/corpus_extended.csv")):
        if r["source"] != "llm-perf-leaderboard":
            continue
        m = r["model"].strip().lower()
        if m not in scores:
            continue
        try:
            t = float(r["latency_ms"])
        except ValueError:
            continue
        if t <= 0:
            continue
        plat = r["cell"].split("|")[2].strip()
        for (label, ch), Q in scores[m].items():
            rows.append(dict(
                task=f"LLM {label}", dataset=label, model=r["model"],
                device=plat, runtime=r["runtime"], precision=r["precision"],
                cell=f"llm-perf | {plat} | {label}", metric=label,
                Q=round(Q, 6), chance=ch, latency_ms=t,
                param_count=r["param_count"], gmacs="", source="llm-perf-task"))
            n += 1
    return n


def source_lib(scores):
    n, seen = 0, set()
    for f in ("lib_all_results.csv", "lib_ppl_fig10.csv", "lib_ppl_fig29.csv"):
        p = f"{DEST}/{f}"
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            m = r["Model"].strip().lower()
            if m not in scores:
                continue
            io, bs = r["Input Output Length"], r["Batch Size"]
            if not LIB_ALL and (io != LIB_IO or bs != LIB_BATCH):
                continue
            try:
                lat = float(r["Latency"]) * 1000.0
            except (TypeError, ValueError):
                continue
            if lat <= 0:
                continue
            hw, fw, ng = r["Hardware"], r["Framework"], r["Num of Hardware"]
            key = (hw, ng, fw, io, bs, m)
            if key in seen:
                continue
            seen.add(key)
            dev = f"{hw} x{ng}"
            tag = "" if not LIB_ALL else f" | io{io} b{bs}"
            for (label, ch), Q in scores[m].items():
                rows.append(dict(
                    task=f"LLM {label}", dataset=label, model=r["Model"],
                    device=dev, runtime=fw, precision="fp16",
                    cell=f"llm-bench | {dev} | {fw}{tag} | {label}",
                    metric=label, Q=round(Q, 6), chance=ch,
                    latency_ms=round(lat, 4), param_count=0, gmacs="",
                    source="llm-inference-bench"))
                n += 1
    return n


def _md_rows(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and line.count("|") > 2:
            yield [c.strip() for c in line.strip("|").split("|")]


def source_llamacpp():
    pp, qq = (f"{DEST}/llamacpp_perplexity_README.md",
              f"{DEST}/llamacpp_quantize_README.md")
    if not (os.path.exists(pp) and os.path.exists(qq)):
        return 0
    ppl = {}
    for c in _md_rows(open(pp).read()):
        if len(c) != 8:
            continue
        q, imat, size, PPL = c[0], c[1], c[2], c[3]
        if not re.fullmatch(r"i?q\d[_a-zA-Z0-9]*|f16", q, re.I):
            continue
        try:
            size, PPL = float(size), float(PPL.split("\u00b1")[0])
        except ValueError:
            continue
        ppl.setdefault(q.lower(), []).append((imat, size, PPL))

    speed, block = {}, []
    for c in _md_rows(open(qq).read()):
        if c and c[0] == "Measure":
            block = [c]
            continue
        if not block or c[0].startswith("---"):
            continue
        if c[0] in ("bits/weight", "size (GiB)") or c[0].startswith(
                ("prompt processing", "text generation")):
            block.append(c)
            if len(block) == 5:
                head = block[0]
                for j in range(1, len(head)):
                    try:
                        speed[head[j].lower()] = dict(
                            size=float(block[2][j]),
                            pp=float(block[3][j].split("\u00b1")[0]),
                            tg=float(block[4][j].split("\u00b1")[0]))
                    except (ValueError, IndexError):
                        pass
                block = []
        else:
            block = []

    n = 0
    for q, entries in sorted(ppl.items()):
        if q not in speed:
            continue
        s = speed[q]
        for imat, size, PPL in entries:
            if abs(size - s["size"]) > 0.05:
                continue
            for kind, tps, tok in (("text generation", s["tg"], 128),
                                   ("prompt processing", s["pp"], 512)):
                rows.append(dict(
                    task="LLM perplexity", dataset="WikiText-2",
                    model=f"Llama-3-8B {q}" + ("" if imat.lower() == "none"
                                               else f" imatrix {imat}"),
                    device="1x RTX 4090", runtime="llama.cpp", precision=q,
                    cell=f"llama.cpp | 1x RTX 4090 | {kind}",
                    metric=f"perplexity/{PPL_REF:g}",
                    Q=round(1 - PPL / PPL_REF, 8), chance=0.0,
                    latency_ms=round(1000.0 * tok / tps, 4),
                    param_count=8, gmacs="", source="llama.cpp"))
                n += 1
    return n


def source_bench360():
    p = f"{DEST}/bench360_table3.csv"
    if not os.path.exists(p):
        return 0
    n = 0
    for r in csv.DictReader(open(p)):
        rows.append(dict(
            task=f"LLM {r['task']}", dataset=r["dataset"], model=r["model"],
            device="24GB-class GPU", runtime="vLLM",
            precision=r["quantisation"],
            cell=f"bench360 | 24GB-class GPU | vLLM | {r['task']}",
            metric=r["metric"], Q=float(r["score"]), chance=float(r["chance"]),
            latency_ms=float(r["tpot_ms"]), param_count=0, gmacs="",
            source="bench360"))
        n += 1
    return n


def main():
    scores = load_scores()
    a = source_llmperf(scores)
    b = source_lib(scores)
    c = source_llamacpp()
    d = source_bench360()
    with open(f"{OUT}/corpus_llm.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    cells = sorted({r["cell"] for r in rows})
    print(f"leaderboard scores for {len(scores)} models over "
          f"{len(V1_TASKS) + len(V2_TASKS)} tasks")
    print(f"llm-perf (task-resolved)   {a:6d} records")
    print(f"llm-inference-bench        {b:6d} records")
    print(f"llama.cpp                  {c:6d} records")
    print(f"bench360                   {d:6d} records")
    print(f"total                      {len(rows):6d} records, "
          f"{len(cells)} candidate cells")
    print(f"wrote {OUT}/corpus_llm.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
