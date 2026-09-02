#!/usr/bin/env python3
"""
finalize.py  --  one estimator, one bootstrap unit, one inference structure.

Earlier scripts mixed three estimators (least squares in epsilon, least squares
in log epsilon), two bootstrap units (frontier points, systems) and one
inference structure that treated nine benchmarks of a single model population as
nine independent observations. This script settles all three.

Estimator.   Least squares in log epsilon, the maximum-likelihood criterion
             under the multiplicative noise the calibration measured. Used
             everywhere below, including inside the calibration and the
             bootstrap, so the bias correction applies to the estimator that
             produces the reported number.

Bootstrap.   Resample the cell's records, recompute the Pareto front, refit.
             What varies between replications of the world is which systems
             exist.

Inference.   The nine long-front cells are the same ~1200 timm checkpoints on
             nine device/runtime configurations. They are nine measurements of
             one population, not nine draws from a population of frontiers.
             Reported accordingly: a device-level interval for this population,
             and an explicit statement that cross-population generalisation
             rests only on the out-of-domain cells.

Writes final_estimate.json, table_final.csv
"""
import collections
import csv
import json
import math
import os
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as tdist

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260824)

_D = os.environ.get("DATA", "data")
_OUT = os.environ.get("OUT", "results")
D, OUT = _D, _OUT
os.makedirs(OUT, exist_ok=True)

MINFRONT = int(os.environ.get("MINFRONT", 20))
NBOOT = int(os.environ.get("NBOOT", 200))
NCAL = int(os.environ.get("NCAL", 40))
NITER = int(os.environ.get("NITER", 3))


def pareto(pts):
    pts = sorted(pts)
    out, best = [], np.inf
    for t, e in pts:
        if e < best - 1e-15:
            out.append((t, e))
            best = e
    return out


def _sig(z, scale):
    z = max(-500.0, min(500.0, float(z)))
    return scale / (1.0 + math.exp(-z))


def _sse(u, lr, le, hi):
    ei, lc, a = _sig(u[0], hi), u[1], _sig(u[2], 8.0)
    m = ei + math.exp(lc) * np.exp(-a * lr)
    if np.any(m <= 0) or not np.all(np.isfinite(m)):
        return 1e30
    return float(((le - np.log(m)) ** 2).sum())


def fit_log(t, e, seed=None):
    """ML fit under multiplicative noise. seed = (eps_inf, c, alpha)."""
    lr, le = np.log(np.asarray(t, float)), np.log(np.asarray(e, float))
    hi = float(np.exp(le).min())
    if seed is not None:
        r = min(seed[0] / hi, 0.999999)
        u0 = [math.log(max(r, 1e-9) / max(1 - r, 1e-9)),
              math.log(max(seed[1], 1e-12)),
              math.log(max(seed[2], 1e-6) / max(8 - seed[2], 1e-6))]
        starts = [u0]
    else:
        starts = [[s, 0., a]
                  for s in (-8., -6., -4., -2., -1., 0., 1., 3., 6., 8.)
                  for a in (-3.5, -2.7, -1.6, -0.8, 0.0, 1.0)]
    best = None
    for u0 in starts:
        r = minimize(_sse, u0, args=(lr, le, hi), method="Nelder-Mead",
                     options=dict(maxiter=30000, maxfev=30000,
                                  xatol=1e-11, fatol=1e-13))
        if not np.isfinite(r.fun) or r.fun > 1e29:
            continue
        if best is None or r.fun < best[3]:
            best = (_sig(r.x[0], hi), math.exp(r.x[1]), _sig(r.x[2], 8.0),
                    float(r.fun))
    return best


# ------------------------------------------------------------------ cells ---
cells = collections.defaultdict(list)
for x in csv.DictReader(open(f"{D}/corpus.csv")):
    Q, ch, t = float(x["Q"]), float(x["chance"]), float(x["latency_ms"])
    if t <= 0:
        continue
    e = 1 - (Q - ch) / (1 - ch)
    if 0 < e < 1:
        cells[(x["source"], x["cell"])].append((t, e))

ref = {}
for k, v in cells.items():
    P = pareto(v)
    if len(P) < MINFRONT:
        continue
    f = fit_log([p[0] for p in P], [p[1] for p in P])
    if f is None or not (0.01 < f[2] < 8):
        continue
    ref[k] = dict(P=P, recs=np.array(v), eps_inf=f[0], c=f[1], alpha=f[2],
                  lam=f[2] / 2)
print(f"cells with fronts >= {MINFRONT}: {len(ref)}   "
      f"(all from the {len(set(k[0] for k in ref))} source family "
      f"{sorted(set(k[0] for k in ref))})\n")

rows = []
for k, d in sorted(ref.items()):
    recs = d["recs"]
    name = str(k[1])[:46]

    # ---- system-level bootstrap under the same estimator
    lams = []
    for _ in range(NBOOT):
        idx = RNG.integers(0, len(recs), len(recs))
        P = pareto(list(map(tuple, recs[idx])))
        if len(P) < MINFRONT:
            continue
        f = fit_log([p[0] for p in P], [p[1] for p in P],
                    seed=(d["eps_inf"], d["c"], d["alpha"]))
        if f and 0.01 < f[2] < 8:
            lams.append(f[2] / 2)
    lams = np.array(lams)
    se = float(lams.std(ddof=1)) if len(lams) > 30 else float("nan")
    lo, hi = (float(np.percentile(lams, 2.5)),
              float(np.percentile(lams, 97.5))) if len(lams) > 30 else (np.nan,) * 2

    # ---- fixed-point bias calibration of the same estimator
    t_all = recs[:, 0]
    pred = d["eps_inf"] + d["c"] * t_all ** (-d["alpha"])
    sigma = float(np.std(np.log(np.clip(recs[:, 1], 1e-9, None) /
                                np.clip(pred, 1e-9, None))))
    lt, ht = float(t_all.min()), float(t_all.max())
    scale = 1.0
    for _ in range(NITER):
        a_true = d["alpha"] / scale
        ah = []
        for _ in range(NCAL):
            t = np.exp(RNG.uniform(math.log(lt), math.log(ht), len(recs)))
            e = np.clip((d["eps_inf"] + d["c"] * t ** (-a_true)) *
                        np.exp(RNG.normal(0, sigma, len(recs))), 1e-6, .999)
            P = pareto(list(zip(t, e)))
            if len(P) < MINFRONT:
                continue
            f = fit_log([p[0] for p in P], [p[1] for p in P],
                        seed=(d["eps_inf"], d["c"], a_true))
            if f and 0.01 < f[2] < 8:
                ah.append(f[2])
        if not ah:
            break
        b = float(np.mean(ah)) / a_true - 1
        if abs((1 + b) - scale) < 0.01:
            scale = 1 + b
            break
        scale = 1 + b
    bias = scale - 1

    rows.append(dict(cell=name, n_front=len(d["P"]), n=len(recs), sigma=sigma,
                     eps_inf=d["eps_inf"], alpha=d["alpha"], lam=d["lam"],
                     boot_lo=lo, boot_hi=hi, boot_se=se, bias=bias,
                     lam_corrected=d["lam"] / (1 + bias), n_boot=len(lams)))
    print(f"{name:46s} lam*={d['lam']:.3f} "
          f"[{lo:.3f},{hi:.3f}] se={se:.3f}  bias={100*bias:+5.1f}%  "
          f"-> {d['lam']/(1+bias):.3f}")

lam = np.array([r["lam"] for r in rows])
lamc = np.array([r["lam_corrected"] for r in rows])
ses = np.array([r["boot_se"] for r in rows])
k = len(rows)

# ---------------------------------------------------- cluster-aware summary --
mean_c = float(lamc.mean())
sd_dev = float(lamc.std(ddof=1))
se_dev = sd_dev / math.sqrt(k)
tc = float(tdist.ppf(0.975, k - 1))
ci_dev = [mean_c - tc * se_dev, mean_c + tc * se_dev]

# the naive pooled interval, for comparison only
w = 1 / np.maximum(ses ** 2, 1e-12)
naive_mu = float((w * lamc).sum() / w.sum())
naive_se = float(math.sqrt(1 / w.sum()))

print("\n" + "=" * 72)
print("FINAL ESTIMATE  (log-space ML estimator, system-level bootstrap,")
print("                 bias-corrected, device-level inference)")
print(f"  population      : timm ImageNet checkpoints, {k} device/runtime cells")
print(f"  mean lambda*    : {mean_c:.3f}")
print(f"  device-to-device sd : {sd_dev:.3f}   SE of mean {se_dev:.3f}")
print(f"  95% interval    : [{ci_dev[0]:.3f}, {ci_dev[1]:.3f}]   "
      f"(t on {k-1} df)")
print(f"  naive pooled interval, for contrast: "
      f"[{naive_mu-1.96*naive_se:.3f}, {naive_mu+1.96*naive_se:.3f}] "
      f"-- {(tc*se_dev)/(1.96*naive_se):.1f}x narrower, and wrong, because it")
print( "  treats nine benchmarks of one population as nine populations.")
print(f"  median bias of the estimator: {100*np.median([r['bias'] for r in rows]):+.1f}%")
print("=" * 72)

with open(f"{OUT}/table_final.csv", "w", newline="") as fh:
    w2 = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w2.writeheader()
    for r in rows:
        w2.writerow(r)
json.dump(dict(estimator="least squares in log epsilon (ML, multiplicative noise)",
               bootstrap="system-level (resample records, recompute front)",
               inference="device-level t interval within one model population",
               k_cells=k, cells=rows,
               mean=mean_c, sd_devices=sd_dev, se=se_dev, ci=ci_dev,
               naive_pooled=naive_mu,
               naive_ci=[naive_mu - 1.96 * naive_se, naive_mu + 1.96 * naive_se],
               median_bias=float(np.median([r["bias"] for r in rows])),
               minfront=MINFRONT, nboot=NBOOT, ncal=NCAL),
          open(f"{OUT}/final_estimate.json", "w"), indent=1)
print("\nwrote final_estimate.json, table_final.csv")
