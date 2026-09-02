#!/usr/bin/env python3
"""
fetch_llm_sources.py  --  download the public, machine-readable LLM inference
benchmarks used by build_llm_corpus.py.

Three independent sources, all served from GitHub so that the fetch is
reproducible without an account:

  A  argonne-lcf/LLM-Inference-Bench      measured latency and throughput for
     15 open-weight models on 9 accelerators under 6 inference frameworks,
     plus a WikiText perplexity for each model measured by the same study.

  B  ggml-org/llama.cpp                   the LLaMA-3-8B quantisation
     scoreboard (perplexity and KL divergence for 30 quantisation types on one
     RTX 4090) and the matching generation- and prompt-speed table.

  C  Bench360 (arXiv:2511.16682) Table 3  four NLP tasks with declared chance
     levels and measured time per output token under a 24 GB budget.  The
     table is transcribed in bench360_table3.csv because the upstream
     repository ships the harness but not the run reports.

Writes everything into data/.  Files already present are left alone, so the
script is safe to re-run and works offline once the archive is populated.
"""
import io
import os
import sys
import urllib.request
import zipfile

DEST = os.environ.get("DATA", "data")
os.makedirs(DEST, exist_ok=True)

RAW = "https://raw.githubusercontent.com"
ZIP = "https://codeload.github.com"

FILES = [
    # (destination, url)
    (f"{DEST}/llamacpp_quantize_README.md",
     f"{RAW}/ggml-org/llama.cpp/master/tools/quantize/README.md"),
    (f"{DEST}/llamacpp_perplexity_README.md",
     f"{RAW}/ggml-org/llama.cpp/master/tools/perplexity/README.md"),
]

# LLM-Inference-Bench keeps its result tables inside the repository tree; the
# individual raw paths contain spaces, so the archive is easier to fetch whole.
LIB_ZIP = f"{ZIP}/argonne-lcf/LLM-Inference-Bench/zip/refs/heads/main"
LIB_MEMBERS = {
    "LLM-Inference-Bench-main/Plots/All_results.csv": f"{DEST}/lib_all_results.csv",
    "LLM-Inference-Bench-main/Plots/Fig_10/All_results.csv": f"{DEST}/lib_ppl_fig10.csv",
    "LLM-Inference-Bench-main/Plots/Fig_29/All_results.csv": f"{DEST}/lib_ppl_fig29.csv",
}


def get(url, dest):
    if os.path.exists(dest):
        print(f"  have {dest}")
        return
    print(f"  get  {dest}")
    with urllib.request.urlopen(url, timeout=120) as r:
        open(dest, "wb").write(r.read())


def main():
    for dest, url in FILES:
        get(url, dest)

    if not all(os.path.exists(v) for v in LIB_MEMBERS.values()):
        print("  get  LLM-Inference-Bench archive")
        with urllib.request.urlopen(LIB_ZIP, timeout=300) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for member, dest in LIB_MEMBERS.items():
                open(dest, "wb").write(z.read(member))
                print(f"       -> {dest}")
    else:
        print("  have LLM-Inference-Bench tables")

    missing = [p for p in
               [d for d, _ in FILES] + list(LIB_MEMBERS.values()) +
               [f"{DEST}/bench360_table3.csv"]
               if not os.path.exists(p)]
    if missing:
        print("MISSING:", *missing, sep="\n  ")
        return 1
    print("all sources present in", DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
