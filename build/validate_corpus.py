#!/usr/bin/env python3
"""Check rebuilt cells against the published Table 5."""
import pandas as pd, numpy as np

def uhull(pf, x, y):
    pf = pf.sort_values(x); pts = pf[[x, y]].values; h = []
    for j, p in enumerate(pts):
        while len(h) >= 2:
            (x1, y1), (x2, y2) = h[-2][:2], h[-1][:2]
            if (y2-y1)*(p[0]-x1) <= (p[1]-y1)*(x2-x1)+1e-12: h.pop()
            else: break
        h.append((p[0], p[1], pf.index[j]))
    return pf.loc[[t[2] for t in h]]

def pareto(d, x, y):
    s = d.sort_values([x, y], ascending=[True, False]); b, k = -np.inf, []
    for i, r in s.iterrows():
        if r[y] > b + 1e-12: k.append(i); b = r[y]
    return d.loc[k].sort_values(x)

PAPER = {
 "timm | CPU i7-12700H | PyTorch 2.4.0 compile | fp32":     (1091,41,13,0.220),
 "timm | CPU i9-10940X | PyTorch 2.4.0 compile | fp32":     (1086,36,10,0.298),
 "timm | RTX 3090 | PyTorch 2.4.0 eager | amp-fp16":        (1091,37, 7,0.238),
 "timm | RTX 4090 | PyTorch 2.9.1 compile | amp-fp16":      (1191,40,10,0.237),
 "timm | RTX 4090 | PyTorch 2.9.1 eager | amp-fp16":        (1193,33, 6,0.236),
 "timm | RTX 5090 | PyTorch 2.9.1 compile | amp-fp16":      (1193,42, 9,0.234),
 "timm | RTX 5090 | PyTorch 2.9.1 eager | amp-fp16":        (1193,34,10,0.232),
 "timm | RTX Pro 6000 | PyTorch 2.9.1 compile | bf16":      (1187,36, 8,0.225),
 "timm | RTX Pro 6000 | PyTorch 2.9.1 eager | amp-fp16":    (1193,36, 5,0.235),
}
ULTRA = {
 ("Image classification","CPU ONNX"):(10,5,4,0.180),
 ("Image classification","NVIDIA T4"):(5,5,4,0.307),
 ("Instance segmentation","CPU ONNX"):(10,5,4,0.082),
 ("Instance segmentation","NVIDIA T4"):(5,5,4,0.088),
 ("Object detection","CPU ONNX"):(20,7,5,0.089),
 ("Object detection","NVIDIA T4"):(10,10,6,0.146),
 ("Oriented detection","CPU ONNX"):(10,6,5,0.054),
 ("Oriented detection","NVIDIA T4"):(5,5,4,0.077),
 ("Pose estimation","CPU ONNX"):(11,6,5,0.130),
 ("Pose estimation","NVIDIA T4"):(5,5,4,0.252),
}

C = pd.read_csv("corpus.csv")
C = C[~((C.task == "Object detection") & C.model.str.contains("-seg", na=False))]

def stats(d):
    d = d.copy()
    d["Qp"] = (d.Q - d.chance) / (1 - d.chance)
    d = d[(d.Qp > 0) & (d.Qp < 1)]
    d["S"] = -np.log(1 - d.Qp); d["lnr"] = np.log(d.latency_ms)
    pf = pareto(d, "lnr", "S"); Ev = uhull(pf, "lnr", "S")
    eta = (Ev.S.iloc[-1]-Ev.S.iloc[0])/(Ev.lnr.iloc[-1]-Ev.lnr.iloc[0])
    return len(d), len(pf), len(Ev), eta

ok = tot = 0
print(f"{'cell':<52}{'n':<11}{'|P|':<9}{'|E|':<8}{'eta':<16}")
print("-"*98)
for cell, exp in PAPER.items():
    got = stats(C[C.cell == cell]); tot += 1
    m = (got[0],got[1],got[2],round(got[3],3)) == exp; ok += m
    print(f"{cell[:50]:<52}{got[0]:>4}/{exp[0]:<5}{got[1]:>3}/{exp[1]:<4}"
          f"{got[2]:>3}/{exp[2]:<3}{got[3]:.3f}/{exp[3]:.3f}  {'OK' if m else '**'}")
for (t, dv), exp in ULTRA.items():
    sub = C[(C.task == t) & (C.device == dv)]
    tot += 1
    if len(sub) == 0:
        print(f"{t+' / '+dv:<52}  not rebuilt"); continue
    got = stats(sub)
    m = (got[0],got[1],got[2],round(got[3],3)) == exp; ok += m
    print(f"{(t+' / '+dv)[:50]:<52}{got[0]:>4}/{exp[0]:<5}{got[1]:>3}/{exp[1]:<4}"
          f"{got[2]:>3}/{exp[2]:<3}{got[3]:.3f}/{exp[3]:.3f}  {'OK' if m else '**'}")
print(f"\nexact matches: {ok}/{tot} rebuilt cells "
      f"({tot} of 25 published cells attempted)")
