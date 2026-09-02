#!/usr/bin/env python3
"""
build_corpus.py -- assemble the cross-domain inference corpus (corpus.csv).

RECONSTRUCTION NOTICE
---------------------
This script was reconstructed from public sources to replace a corpus file
that was not archived with the original analysis. It is NOT the original
extraction code. It reproduces 18 of the 25 device-runtime cells of the
published Table 5 exactly (n, |P|, |E| and eta_bar all agree to the printed
precision). The remaining cells are documented in RECONSTRUCTION.md.

Usage:
    python fetch_sources.py     # downloads public source data
    python build_corpus.py      # writes corpus.csv
    python validate_corpus.py   # checks cells against published Table 5
"""
import subprocess, sys, pandas as pd

def main():
    for step in ("build_timm.py", "build_ultra.py"):
        print(f"--- running {step}")
        subprocess.check_call([sys.executable, step])
    A = pd.read_csv("corpus_imagenet.csv")
    B = pd.read_csv("corpus_ultralytics.csv")
    C = pd.concat([A, B], ignore_index=True)
    cols = ["task","dataset","model","device","runtime","precision","cell",
            "metric","Q","chance","latency_ms","param_count","gmacs","source"]
    C = C[[c for c in cols if c in C.columns]]
    C.to_csv("corpus.csv", index=False)
    print(f"\nwrote corpus.csv: {len(C)} rows, {C.cell.nunique()} cells, "
          f"{C.task.nunique()} tasks")

if __name__ == "__main__":
    main()
