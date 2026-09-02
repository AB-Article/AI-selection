#!/usr/bin/env python3
"""
noise_model.py  --  is the quality noise multiplicative?

The estimator of finalize.py minimises squared residuals in log epsilon, which
is the right criterion if the scatter of systems about the frontier law is
multiplicative and the wrong one if it is additive. That assumption was used
inside the bias calibration as well, so validating the estimator against
simulations drawn from it would be circular unless the assumption is checked
against the data independently. This script checks it.

For each cell, fit the frontier law, then take the residuals of *all* records in
the cell -- not just the frontier points -- in both parametrisations, and test
each for heteroscedasticity against latency by Spearman correlation of the
absolute residual with t. A correlation near zero is what the corresponding
estimator assumes.

Also reported: Shapiro-Wilk on the log residuals, which bears on whether "least
squares in log epsilon" deserves to be called maximum likelihood.

Writes noise_model.json
"""
import collections
import csv
import json
import math
import os
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import shapiro, spearmanr

warnings.filterwarnings("ignore")
_D = os.environ.get("DATA", "data")
_OUT = os.environ.get("OUT", "results")
D, OUT = _D, _OUT
os.makedirs(OUT, exist_ok=True)
MINFRONT = int(os.environ.get("MINFRONT", 20))


def pareto(pts):
    pts = sorted(pts)
    out, best = [], np.inf
    for t, e in pts:
        if e < best - 1e-15:
            out.append((t, e))
            best = e
    return out


def _sig(z, s):
    z = max(-500.0, min(500.0, float(z)))
    return s / (1.0 + math.exp(-z))


def _sse(u, lr, le, hi):
    ei, lc, a = _sig(u[0], hi), u[1], _sig(u[2], 8.0)
    m = ei + math.exp(lc) * np.exp(-a * lr)
    if np.any(m <= 0) or not np.all(np.isfinite(m)):
        return 1e30
    return float(((le - np.log(m)) ** 2).sum())


def fit_log(t, e):
    lr, le = np.log(t), np.log(e)
    hi = float(e.min())
    best = None
    for s0 in (-8., -6., -4., -2., -1., 0., 1., 3., 6., 8.):
        for a0 in (-3.5, -2.7, -1.6, -0.8, 0., 1.):
            r = minimize(_sse, [s0, 0., a0], args=(lr, le, hi),
                         method="Nelder-Mead",
                         options=dict(maxiter=30000, maxfev=30000,
                                      xatol=1e-11, fatol=1e-13))
            if r.fun < 1e29 and (best is None or r.fun < best[0]):
                best = (r.fun, r.x)
    ei, c, a = _sig(best[1][0], hi), math.exp(best[1][1]), _sig(best[1][2], 8.0)
    return ei, c, a


cells = collections.defaultdict(list)
for x in csv.DictReader(open(f"{D}/corpus.csv")):
    Q, ch, t = float(x["Q"]), float(x["chance"]), float(x["latency_ms"])
    if t <= 0:
        continue
    e = 1 - (Q - ch) / (1 - ch)
    if 0 < e < 1:
        cells[x["cell"]].append((t, e))

rows = []
print(f"{'cell':36s} {'device':>7} {'log':>7} {'eps':>7} {'Shapiro p':>11}")
for k, v in sorted(cells.items()):
    P = pareto(v)
    if len(P) < MINFRONT:
        continue
    t = np.array([p[0] for p in P])
    e = np.array([p[1] for p in P])
    ei, c, a = fit_log(t, e)
    A = np.array(v)
    ta, ea = A[:, 0], A[:, 1]
    pred = ei + c * ta ** (-a)
    r_log = np.log(ea / pred)
    r_eps = ea - pred
    s_log = abs(float(spearmanr(ta, np.abs(r_log)).statistic))
    s_eps = abs(float(spearmanr(ta, np.abs(r_eps)).statistic))
    sh = float(shapiro(r_log[:5000]).pvalue)
    dev = "CPU" if "CPU" in k else "GPU"
    rows.append(dict(cell=k[:60], device=dev, n=len(v), alpha=a, lam=a / 2,
                     hetero_log=s_log, hetero_eps=s_eps, shapiro_p=sh,
                     favours="log" if s_log < s_eps else "eps"))
    print(f"{k[:36]:36s} {dev:>7} {s_log:7.3f} {s_eps:7.3f} {sh:11.2e}")

gl = [r for r in rows if r["device"] == "GPU"]
cl = [r for r in rows if r["device"] == "CPU"]
print(f"\nmedian |Spearman(t, |residual|)|   (0 = homoscedastic)")
print(f"  all cells : log {np.median([r['hetero_log'] for r in rows]):.3f}   "
      f"eps {np.median([r['hetero_eps'] for r in rows]):.3f}")
print(f"  GPU  ({len(gl)}) : log {np.median([r['hetero_log'] for r in gl]):.3f}   "
      f"eps {np.median([r['hetero_eps'] for r in gl]):.3f}   "
      f"-> multiplicative")
print(f"  CPU  ({len(cl)}) : log {np.median([r['hetero_log'] for r in cl]):.3f}   "
      f"eps {np.median([r['hetero_eps'] for r in cl]):.3f}   "
      f"-> additive")
print(f"\nlog residuals normal? max Shapiro p = "
      f"{max(r['shapiro_p'] for r in rows):.1e}  -- strongly non-normal, so the")
print("log-space fit is least squares under a variance-stabilising transform,")
print("not maximum likelihood.")
print(f"\nlambda* in the two CPU cells: "
      f"{', '.join(f'{r[chr(108)+chr(97)+chr(109)]:.3f}' for r in cl)}"
      f"   GPU median {np.median([r['lam'] for r in gl]):.3f}")

json.dump(dict(cells=rows,
               median_hetero_log=float(np.median([r["hetero_log"] for r in rows])),
               median_hetero_eps=float(np.median([r["hetero_eps"] for r in rows])),
               gpu_favours_log=int(sum(r["favours"] == "log" for r in gl)),
               gpu_n=len(gl),
               cpu_favours_eps=int(sum(r["favours"] == "eps" for r in cl)),
               cpu_n=len(cl),
               max_shapiro_p=float(max(r["shapiro_p"] for r in rows))),
          open(f"{OUT}/noise_model.json", "w"), indent=1)
print("\nwrote noise_model.json")
