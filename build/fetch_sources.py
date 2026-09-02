#!/usr/bin/env python3
"""Download the public source data needed by build_corpus.py."""
import io, tarfile, urllib.request, pathlib

TIMM = "https://codeload.github.com/huggingface/pytorch-image-models/tar.gz/refs/heads/main"
ULTRA = "https://codeload.github.com/ultralytics/ultralytics/tar.gz/refs/heads/main"

def grab(url, members_prefix, strip):
    print("downloading", url)
    raw = urllib.request.urlopen(url).read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
        for m in t.getmembers():
            if members_prefix in m.name and m.isfile():
                m.name = "/".join(m.name.split("/")[strip:])
                t.extract(m, ".")

grab(TIMM, "/results/", 2)                       # -> ./*.csv
grab(ULTRA, "/docs/en/", 0)                      # -> ./ultralytics-main/docs/en/**
print("done")
