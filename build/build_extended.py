#!/usr/bin/env python3
"""
build_extended.py  --  assemble the extended cross-domain corpus.

Four public sources, each of which publishes a quality metric and a measured
inference cost for many systems evaluated under one fixed condition:

  mmsegmentation   semantic segmentation, mIoU, "Inf time (fps)" on V100/A100
  mmdetection      object detection and instance segmentation, box/mask AP,
                   "Inf time (fps)" on V100
  Open ASR         speech recognition, word error rate and RTFx (inverse
  Leaderboard      real-time factor) on the leaderboard's fixed GPU
  LLM-Perf         open-weight LLM inference, Open LLM Leaderboard score and
  Leaderboard      end-to-end latency on A100 / A10 / T4 / 32-vCPU Xeon

Quality is mapped to a chance-corrected error in (0,1) in every case:

  mIoU, AP              eps = 1 - Q,            chance 0
  word error rate       eps = WER/100,          already an error rate
  Open LLM score        eps = 1 - score/100,    chance 0 (the leaderboard's
                        aggregate is normalised so that a random baseline
                        scores 0)

Resource is a positive number whose logarithm is the paper's x-axis:

  mm* zoos              milliseconds, 1000/fps
  Open ASR              1/RTFx, dimensionless inverse throughput
  LLM-Perf              end-to-end seconds

Theorem 8(ii) makes the choice of unit irrelevant and 8(iii) states how the
price transforms if a resource is a power of another, so mixing units across
cells is legitimate: a price is nats per e-fold of a *declared* resource.

Usage:
    bash fetch_extended.sh          # clone the four sources into ./src
    python3 build_extended.py       # -> corpus_extended.csv
"""
import csv
import glob
import os
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "src"
FIELDS = ["task", "dataset", "model", "device", "runtime", "precision", "cell",
          "metric", "Q", "chance", "latency_ms", "param_count", "gmacs",
          "source"]


# ----------------------------------------------------------------- markdown --
def parse_tables(path):
    section, header = "", None
    for line in open(path, errors="ignore"):
        t = line.strip()
        if t.startswith("#"):
            section, header = t.lstrip("#").strip(), None
            continue
        if not t.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", t.strip("|"))]
        if set("".join(cells)) <= set("-: "):
            continue
        if header is None:
            header = cells
            continue
        if len(cells) >= len(header):
            yield section, header, cells[:len(header)]


def num(x):
    x = re.sub(r"[\\*]", "", x).strip()
    m = re.match(r"^-?\d+(\.\d+)?", x)
    return float(m.group()) if m else None


def col(header, *names):
    low = [h.lower() for h in header]
    for n in names:
        if n.lower() in low:
            return low.index(n.lower())
    return None


def mmlab(rows, repo, root, spec, dataset_override=None):
    for f in sorted(glob.glob(os.path.join(root, "configs/**/README.md"),
                              recursive=True)):
        family = os.path.basename(os.path.dirname(f))
        for section, header, cells in parse_tables(f):
            i = col(header, "Inf time (fps)")
            if i is None:
                continue
            fps = num(cells[i])
            if not fps or fps <= 0:
                continue
            di = col(header, "Device")
            device = (cells[di].strip() if di is not None else "") or "V100"
            if device == "-":
                device = "V100"
            bi, ci = col(header, "Backbone"), col(header, "Crop Size")
            bb = cells[bi] if bi is not None else ""
            cs = cells[ci] if ci is not None else ""
            dataset = dataset_override or section or "unknown"
            for metric, task in spec:
                mi = col(header, metric)
                if mi is None:
                    continue
                q = num(cells[mi])
                if q is None or not (0 < q <= 100):
                    continue
                rows.append(dict(
                    task=task, dataset=dataset,
                    model=re.sub(r"\s+", " ", f"{family} {bb} {cs}").strip(),
                    device=device, runtime=repo, precision="fp32",
                    cell=f"{repo} | {dataset} | {device}", metric=metric,
                    Q=round(q / 100.0, 5), chance=0.0,
                    latency_ms=round(1000.0 / fps, 4),
                    param_count="", gmacs="", source=repo))


# ---------------------------------------------------------------- Open ASR --
def asr(rows, root):
    spec = [("en_shortform", "Avg. WER", "English, short-form"),
            ("en_longform", "Average", "English, long-form"),
            ("multilingual", "Avg", "Multilingual")]
    for fname, qcol, label in spec:
        p = os.path.join(root, "scripts", "data", f"{fname}.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            name = r.get("model") or r.get("model_id") or ""
            try:
                wer, rtfx = float(r[qcol]), float(r["RTFx"])
            except (KeyError, TypeError, ValueError):
                continue
            if rtfx <= 0 or not (0 < wer < 100):
                continue
            rows.append(dict(
                task="Speech recognition", dataset=label, model=name,
                device="leaderboard GPU", runtime="open-asr-leaderboard",
                precision="mixed",
                cell=f"open-asr | {label} | leaderboard GPU",
                metric="WER", Q=round(1 - wer / 100.0, 5), chance=0.0,
                latency_ms=round(1.0 / rtfx, 6),
                param_count=r.get("Model size (B)", ""), gmacs="",
                source="open-asr-leaderboard"))


# ---------------------------------------------------------------- LLM-Perf --
def llmperf(rows, root):
    for p in sorted(glob.glob(os.path.join(root, "data",
                                           "llm-perf-leaderboard-*.csv"))):
        hw = re.sub(r".*llm-perf-leaderboard-(.*)\.csv", r"\1", p)
        for r in csv.DictReader(open(p)):
            try:
                s = float(r["Open LLM Score (%)"])
                t = float(r["End-to-End (s)"])
            except (KeyError, TypeError, ValueError):
                continue
            if t <= 0 or not (0 < s < 100):
                continue
            rows.append(dict(
                task="LLM inference", dataset="Open LLM Leaderboard",
                model=r.get("Model \U0001f917", "") or r.get("Model", ""),
                device=hw, runtime=r.get("Backend \U0001f3ed", "pytorch"),
                precision=r.get("Precision \U0001f4e5", ""),
                cell=f"llm-perf | Open LLM | {hw}", metric="Open LLM score",
                Q=round(s / 100.0, 5), chance=0.0,
                latency_ms=round(t * 1000.0, 4),
                param_count=r.get("Params (B)", ""), gmacs="",
                source="llm-perf-leaderboard"))


def main():
    rows = []
    mmlab(rows, "mmseg", os.path.join(SRC, "mmsegmentation"),
          [("mIoU", "Semantic segmentation")])
    mmlab(rows, "mmdet", os.path.join(SRC, "mmdetection"),
          [("box AP", "Object detection"), ("mask AP", "Instance segmentation")],
          dataset_override="COCO")
    asr(rows, os.path.join(SRC, "open_asr_leaderboard"))
    llmperf(rows, os.path.join(SRC, "llm-perf-leaderboard-snapshot"))

    best = {}
    for r in rows:
        k = (r["cell"], r["task"], r["model"], r["latency_ms"])
        if k not in best or r["Q"] > best[k]["Q"]:
            best[k] = r
    rows = sorted(best.values(),
                  key=lambda z: (z["source"], z["cell"], z["task"],
                                 z["latency_ms"]))

    with open("corpus_extended.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    from collections import Counter
    c = Counter((r["source"], r["task"], r["cell"]) for r in rows)
    print(f"{len(rows)} records, {len(c)} (source, task, cell) groups")
    for k, v in sorted(c.items(), key=lambda z: -z[1]):
        if v >= 6:
            print(f"  {v:5d}  {k[1]:24s} {k[2]}")


if __name__ == "__main__":
    main()
