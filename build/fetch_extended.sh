#!/usr/bin/env bash
# fetch_extended.sh -- clone the four public sources behind corpus_extended.csv.
#
# All four are living repositories. corpus_extended.csv is deposited alongside
# this script so that the numbers in the paper remain reproducible even after
# the upstream sources change; re-running the pipeline against a later state of
# these repositories will give a slightly different corpus.
set -eu
mkdir -p src
cd src

clone () {  # clone <url> <dir>
  if [ -d "$2" ]; then echo "have $2"; else
    echo "cloning $2"; git clone -q --depth 1 "$1" "$2"
  fi
}

# semantic segmentation: mIoU and "Inf time (fps)" on V100 / A100 / TITAN Xp
clone https://github.com/open-mmlab/mmsegmentation.git mmsegmentation

# object and instance detection: box/mask AP and "Inf time (fps)" on V100
clone https://github.com/open-mmlab/mmdetection.git   mmdetection

# speech recognition: word error rate and RTFx
clone https://github.com/huggingface/open_asr_leaderboard.git open_asr_leaderboard

# open-weight LLM inference: Open LLM score and end-to-end latency.
# NOTE this is a public snapshot of the upstream HuggingFace dataset
# optimum-benchmark/llm-perf-leaderboard, which is not reachable in
# machine-readable form from every environment. Treat it as a mirror; the
# paper flags the provenance limitation explicitly.
clone https://github.com/dedanimalfarm/llm-perf-leaderboard-snapshot.git \
      llm-perf-leaderboard-snapshot

cd ..
echo
echo "next:  python3 build_extended.py"
