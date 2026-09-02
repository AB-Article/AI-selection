#!/usr/bin/env python3
"""
build_qwen.py -- two more families for the measured-latency ladders.

The Qwen1.5 ladders of Section 8.4.5 rest on one model family, because LLM-Perf
covers only one densely enough.  Two further families can be assembled from
material their authors published: Qwen release their own speed benchmark, at
batch one, on fixed hardware, for every released size, and the Qwen3 technical
report tabulates base-model scores for every size of both Qwen2.5 and Qwen3 on a
common set of benchmarks.

  Qwen2.5  7 sizes, 0.5B-72B, A100 80GB, Transformers and vLLM
  Qwen3    6 dense sizes, 0.6B-32B, H20 96GB, Transformers and SGLang

Speed is reported as (prompt + generated) tokens per second for a 2048-token
generation, so the latency of that generation is (input_len + 2048) / speed.
The speed tables are measured on the instruction-tuned checkpoints and the
quality tables on the base checkpoints; the two share an architecture and a
parameter count at every size, and latency at batch one is a property of the
architecture, so they are joined on size.  That join is the one assumption in
this file and it is recorded in the manuscript.

Writes  corpus_qwen.csv
"""
import csv
import os
import re
import sys
import urllib.request

OUT = os.environ.get("OUT", "results")
Q3_TXT = os.environ.get("Q3_TXT", os.path.join(OUT, "_qwen3_report.txt"))
SRC_Q25 = ("https://raw.githubusercontent.com/QwenLM/Qwen2.5/v2.5/"
           "docs/source/benchmark/speed_benchmark.rst")
SRC_Q3 = ("https://raw.githubusercontent.com/QwenLM/Qwen3/main/"
          "docs/source/getting_started/speed_benchmark.md")

PARAMS = {  # billions, as released
    "Qwen2.5-0.5B": 0.49, "Qwen2.5-1.5B": 1.54, "Qwen2.5-3B": 3.09,
    "Qwen2.5-7B": 7.62, "Qwen2.5-14B": 14.77, "Qwen2.5-32B": 32.76,
    "Qwen2.5-72B": 72.71,
    "Qwen3-0.6B": 0.60, "Qwen3-1.7B": 1.72, "Qwen3-4B": 4.02,
    "Qwen3-8B": 8.19, "Qwen3-14B": 14.77, "Qwen3-32B": 32.76,
}
# benchmarks retired from the Open LLM Leaderboard as models reached them,
# against those kept or introduced to preserve headroom.  Fixed before fitting.
CLASS = {"MMLU": "retired", "MMLU-Redux": "retired", "GSM8K": "retired",
         "BBH": "headroom", "MMLU-Pro": "headroom", "GPQA": "headroom",
         "SuperGPQA": "headroom", "MATH": "headroom", "EvalPlus": "headroom",
         "MultiPL-E": "headroom", "MBPP": "headroom", "CRUX-O": "headroom",
         "MGSM": "retired", "MMMLU": "retired", "INCLUDE": "headroom"}
CHANCE = {"MMLU": .25, "MMLU-Redux": .25, "MMLU-Pro": .10, "GPQA": .25,
          "SuperGPQA": .10, "MMMLU": .25}   # multiple choice; else 0


def fetch(url, name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        print(f"fetching {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, p)
    return open(p, encoding="utf-8", errors="replace").read()


# ------------------------------------------------ Qwen2.5 speed, RST grid --
def parse_q25(text):
    out, backend = [], None
    model = inlen = None
    for line in text.splitlines():
        m = re.match(r"^-\s+(\S+)\s+\((Transformer|vLLM)\)", line.strip())
        if m:
            backend = "Transformers" if m.group(2).startswith("Transf") else "vLLM"
            model = inlen = None
            continue
        if not line.startswith("|") or backend is None:
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 5 or c[0].startswith("Model"):
            continue
        if c[0]:
            mm = re.match(r"(Qwen2\.5-[\d.]+B)", c[0])
            if mm:
                model = mm.group(1)
        if c[1]:
            try:
                inlen = int(c[1])
            except ValueError:
                pass
        quant, gpu, speed = c[2], c[3], c[4]
        if quant != "BF16" or not model or inlen is None:
            continue
        try:
            sp, ng = float(speed), int(gpu)
        except ValueError:
            continue
        out.append(dict(family="Qwen2.5", model=model, params=PARAMS.get(model),
                        device="A100 80GB", backend=backend, precision="BF16",
                        input_len=inlen, gpus=ng, speed_tps=sp,
                        latency_ms=1000 * (inlen + 2048) / sp))
    return out


# ------------------------------------------------- Qwen3 speed, HTML table --
def parse_q3(text):
    out, model, backend, inlen = [], None, None, None
    for line in text.replace("\r", "").splitlines():
        h = re.match(r"^###\s+(Qwen3-[\dA-Za-z.\-]+)\s+\((SGLang|Transformers)\)",
                     line.strip())
        if h:
            model, backend, inlen = h.group(1), h.group(2), None
            continue
        if model is None or "<td" not in line:
            continue
        cells = re.findall(r"<td[^>]*>(.*?)</td>", line)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        cells = [c for c in cells if not c.startswith("Qwen3-")]
        nums = [c for c in cells if c]
        if len(nums) >= 4 and re.fullmatch(r"\d+", nums[0]):
            inlen = int(nums[0])
            nums = nums[1:]
        if len(nums) < 3 or inlen is None:
            continue
        quant, gpu, speed = nums[0], nums[1], nums[2]
        if quant != "BF16":
            continue
        try:
            sp, ng = float(speed), int(gpu)
        except ValueError:
            continue
        if model not in PARAMS:        # skip the mixture-of-experts sizes
            continue
        out.append(dict(family="Qwen3", model=model, params=PARAMS[model],
                        device="H20 96GB", backend=backend, precision="BF16",
                        input_len=inlen, gpus=ng, speed_tps=sp,
                        latency_ms=1000 * (inlen + 2048) / sp))
    return out


# --------------------------------------- quality, Qwen3 technical report ----
def parse_quality(text):
    """Read the per-size base-model tables; columns are named in the header."""
    scores, cols = {}, None
    for line in text.splitlines():
        names = re.findall(r"(Qwen2\.5-[\d.]+B|Qwen3-[\d.]+B)(?![\w-])", line)
        toks = re.split(r"\s{2,}", line.strip())
        # a header line names models; a data line carries two-decimal scores
        if len(names) >= 2 and not re.search(r"\b\d+\.\d{2}\b", line):
            # adjacent columns can end up separated by a single space in the
            # extracted layout; split those apart before matching
            flat = []
            for t in toks:
                parts = t.split() if len(t.split()) > 1 and len(
                    re.findall(r"(?:Qwen2\.5|Qwen3|Gemma|Llama|DeepSeek)[\w.\-]*",
                               t)) > 1 else [t]
                flat += parts
            cols = []
            for t in flat:
                m = re.fullmatch(r"(Qwen2\.5-[\d.]+B|Qwen3-[\d.]+B)(-Base)?", t)
                cols.append(m.group(1) if m else None)
            continue
        if cols is None:
            continue
        m = re.match(r"^\s*([A-Za-z][\w\-. ]*?)\s{2,}([\d.\s]+)$", line)
        if not m:
            continue
        metric = m.group(1).strip()
        if metric not in CLASS:
            continue
        vals = re.findall(r"\d+\.\d+", m.group(2))
        if len(vals) != len(cols):
            continue
        for name, v in zip(cols, vals):
            if name and name in PARAMS:
                scores.setdefault((name, metric), float(v) / 100.0)
    return scores


speed = parse_q25(fetch(SRC_Q25, "_qwen25_speed.rst")) + \
        parse_q3(fetch(SRC_Q3, "_qwen3_speed.md"))
quality = parse_quality(open(Q3_TXT, encoding="utf-8", errors="replace").read())

print(f"speed rows: {len(speed)}  "
      f"({len({(r['family'], r['model']) for r in speed})} model/size pairs)")
print(f"quality: {len(quality)} (model, benchmark) scores over "
      f"{len({k[0] for k in quality})} sizes and "
      f"{len({k[1] for k in quality})} benchmarks")

rows = []
for s in speed:
    for (model, metric), q in quality.items():
        if model != s["model"]:
            continue
        rows.append(dict(s, task=metric, Q=q, chance=CHANCE.get(metric, 0.0),
                         benchmark_class=CLASS[metric]))

cols = ["family", "model", "params", "device", "backend", "precision",
        "input_len", "gpus", "speed_tps", "latency_ms", "task", "Q", "chance",
        "benchmark_class"]
with open(f"{OUT}/corpus_qwen.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda z: (z["family"], z["task"], z["backend"],
                                         z["input_len"], z["params"] or 0)):
        w.writerow(r)
print(f"wrote {OUT}/corpus_qwen.csv: {len(rows)} records, "
      f"{len({(r['family'], r['backend'], r['input_len'], r['task']) for r in rows})} "
      f"candidate cells")
