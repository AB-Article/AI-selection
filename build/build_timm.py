"""Rebuild the timm ImageNet-1k portion of the inference corpus.
Sources: huggingface/pytorch-image-models @ results/ (public, append-only)."""
import pandas as pd, numpy as np

acc = pd.read_csv("results-imagenet.csv")
acc["base"] = acc.model.str.split(".").str[0]
acc = acc.sort_values("top1", ascending=False).drop_duplicates(["base", "img_size"])

CELLS = {
 "benchmark-infer-fp32-nchw-pt240-cpu-i7_12700h-dynamo.csv":  ("CPU i7-12700H","PyTorch 2.4.0 compile","fp32"),
 "benchmark-infer-fp32-nchw-pt240-cpu-i9_10940x-dynamo.csv":  ("CPU i9-10940X","PyTorch 2.4.0 compile","fp32"),
 "benchmark-infer-amp-nchw-pt240-cu124-rtx3090.csv":          ("RTX 3090","PyTorch 2.4.0 eager","amp-fp16"),
 "benchmark-infer-amp-nchw-pt291-cu128-4090-dynamo.csv":      ("RTX 4090","PyTorch 2.9.1 compile","amp-fp16"),
 "benchmark-infer-amp-nchw-pt291-cu128-4090.csv":             ("RTX 4090","PyTorch 2.9.1 eager","amp-fp16"),
 "benchmark-infer-amp-nchw-pt291-cu130-5090-dynamo.csv":      ("RTX 5090","PyTorch 2.9.1 compile","amp-fp16"),
 "benchmark-infer-amp-nchw-pt291-cu130-5090.csv":             ("RTX 5090","PyTorch 2.9.1 eager","amp-fp16"),
 "benchmark-infer-bf16-nchw-pt291-cu130-pro6000maxq-dynamo.csv":("RTX Pro 6000","PyTorch 2.9.1 compile","bf16"),
 "benchmark-infer-amp-nchw-pt291-cu130-pro6000maxq.csv":      ("RTX Pro 6000","PyTorch 2.9.1 eager","amp-fp16"),
}
rows = []
for f, (dev, rt, prec) in CELLS.items():
    b = pd.read_csv(f).rename(columns={"infer_img_size": "img_size"})
    j = b.merge(acc[["base","img_size","top1"]], left_on=["model","img_size"],
                right_on=["base","img_size"])
    j = j[(j.infer_samples_per_sec > 0) & (j.param_count > 0)]
    rows.append(pd.DataFrame(dict(
        task="Image classification", dataset="ImageNet-1k", model=j.model,
        device=dev, runtime=rt, precision=prec,
        cell=f"timm | {dev} | {rt} | {prec}", metric="top1",
        Q=j.top1/100, chance=0.001, latency_ms=1000/j.infer_samples_per_sec,
        param_count=j.param_count, gmacs=j.infer_gmacs, source="timm")))
C = pd.concat(rows, ignore_index=True)
C.to_csv("corpus_imagenet.csv", index=False)
print("timm rows:", len(C), "cells:", C.cell.nunique())
