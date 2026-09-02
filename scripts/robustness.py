#!/usr/bin/env python3
"""
robustness.py  --  five checks that answer the strongest objections to the
empirical claims of this paper.

R1  Winner's curse. A Pareto front is a set of order statistics, so fitting to
    it is fitting to the lucky tail of a noisy population. Simulation with a
    known ground truth quantifies the bias in alpha-hat and the coverage of the
    bootstrap interval, and yields a correction factor.

R2  System-level bootstrap. The front-level bootstrap of stats_meta.py
    resamples frontier points, which is the wrong source of randomness: what
    varies between replications of the world is *which systems exist*, not
    which of the existing ones we look at. Here we resample the cell's records,
    recompute the Pareto front, and refit -- the correct nonparametric
    bootstrap for this design.

R3  Clustered pooling. Nine of the twelve primary cells are the same ~1200 timm
    checkpoints on different hardware, so they are not independent draws.
    Cluster-level pooling treats each source family as one observation.

R4  Identifiability-rule sensitivity. The rule (front >= 6, interior floor) is
    stated in advance but conditions on a fitted quantity. Vary it and see
    whether the pooled price moves.

R5  Metric dependence. The same task and dataset under two quality metrics
    gives two prices, which is the empirical face of the fact that lambda* is
    defined relative to a declared information scale.

Writes robustness.json
"""
import collections
import csv
import json
import math
import os
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t as tdist

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260824)

_D = os.environ.get("DATA", "data")
_OUT = os.environ.get("OUT", "results")
D, OUT = _D, _OUT
os.makedirs(OUT, exist_ok=True)

BSYS = int(os.environ.get("BSYS", 400))   # system-level bootstrap replicates
NSIM = int(os.environ.get("NSIM", 400))   # simulation replicates per setting


# ----------------------------------------------------------------- shared ---
def pareto(pts):
    pts = sorted(pts)
    out, best = [], np.inf
    for t, e in pts:
        if e < best - 1e-15:
            out.append((t, e))
            best = e
    return out


def fit_knee(t, e, seed=None):
    lr, ee = np.log(np.asarray(t, float)), np.asarray(e, float)
    f = lambda lr, ei, lc, a: ei + np.exp(lc) * np.exp(-a * lr)
    hi = min(ee) * .999
    starts = ([[min(seed[0], hi * .9), math.log(max(seed[1], 1e-12)), seed[2]]]
              if seed else
              [[min(ee) * s, 0., .5] for s in (0., .3, .6, .9)])
    best = None
    for p0 in starts:
        try:
            p, _ = curve_fit(f, lr, ee, p0=p0, maxfev=200000,
                             bounds=([0, -40, .01], [hi, 40, 8]))
        except Exception:
            continue
        r = ee - f(lr, *p)
        r2 = 1 - (r ** 2).sum() / max(((ee - ee.mean()) ** 2).sum(), 1e-300)
        if best is None or r2 > best[3]:
            best = (p[0], math.exp(p[1]), p[2], r2)
    return best


def dl_pool(y, s2):
    y, s2 = np.asarray(y, float), np.maximum(np.asarray(s2, float), 1e-12)
    k = len(y)
    if k < 2:
        return float(y[0]), float(math.sqrt(s2[0])), 0.0, [float(y[0])] * 2
    w = 1 / s2
    mu_fe = (w * y).sum() / w.sum()
    Q = (w * (y - mu_fe) ** 2).sum()
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C)
    wr = 1 / (s2 + tau2)
    mu = float((wr * y).sum() / wr.sum())
    se = float(math.sqrt(1 / wr.sum()))
    tc = float(tdist.ppf(0.975, max(k - 2, 1)))
    return mu, se, math.sqrt(tau2), [mu - tc * math.sqrt(tau2 + se ** 2),
                                     mu + tc * math.sqrt(tau2 + se ** 2)]


res = {}


# --- incremental checkpointing -----------------------------------------
# The winner's-curse and calibration simulations are the slow part of this
# script (roughly twenty minutes in total).  Each stage writes the results
# accumulated so far, so an interrupted run still leaves a usable, clearly
# partial robustness.json rather than nothing at all.
def checkpoint(stage):
    res["_stages_completed"] = res.get("_stages_completed", []) + [stage]
    json.dump(res, open(f"{OUT}/robustness.json", "w"), indent=1, default=float)
    print(f"   [checkpoint: {stage}]", flush=True)

# =========================================================== R1 winner's curse
print("R1 winner's curse simulation")
sim_rows = []
for n_sys, sigma in ((1200, 0.10), (1200, 0.20), (1200, 0.35),
                     (200, 0.20), (40, 0.20), (20, 0.20)):
    a_true, ei_true, c_true = 0.70, 0.10, 0.20   # knee at t* = 2.7
    ahat, covered, fronts = [], 0, []
    for _ in range(NSIM):
        # a population of systems: log-uniform cost, mean curve + lognormal
        # quality noise, i.e. each architecture is better or worse than the
        # frontier law by an idiosyncratic factor
        t = np.exp(RNG.uniform(math.log(0.05), math.log(200), n_sys))
        mean_e = ei_true + c_true * t ** (-a_true)
        e = np.clip(mean_e * np.exp(RNG.normal(0, sigma, n_sys)), 1e-6, .999)
        P = pareto(list(zip(t, e)))
        if len(P) < 6:
            continue
        f = fit_knee([p[0] for p in P], [p[1] for p in P])
        if f is None or not (0 < f[2] < 8):
            continue
        if f[0] <= 1e-6 or f[0] >= .999 * min(p[1] for p in P):
            continue
        ahat.append(f[2])
        fronts.append(len(P))
        # front-level bootstrap interval, as used in the paper
        Pa = np.array(P)
        bl = []
        for _b in range(120):
            idx = RNG.integers(0, len(Pa), len(Pa))
            tt, ee2 = Pa[idx, 0], Pa[idx, 1]
            if len(np.unique(tt)) < 4:
                continue
            g = fit_knee(tt, ee2, seed=(f[0], f[1], f[2]))
            if g and 0 < g[2] < 8:
                bl.append(g[2])
        if len(bl) > 30:
            lo, hi = np.percentile(bl, [2.5, 97.5])
            covered += int(lo <= a_true <= hi)
    ahat = np.array(ahat)
    row = dict(n_sys=n_sys, sigma=sigma, n_ok=len(ahat),
               alpha_true=a_true,
               alpha_mean=float(ahat.mean()),
               alpha_median=float(np.median(ahat)),
               rel_bias=float(ahat.mean() / a_true - 1),
               coverage=covered / max(len(ahat), 1),
               front_len=float(np.mean(fronts)))
    sim_rows.append(row)
    print(f"   n={n_sys:5d} sigma={sigma:.2f}  alpha-hat={row['alpha_mean']:.3f} "
          f"(true {a_true})  bias={100*row['rel_bias']:+.1f}%  "
          f"CI coverage={row['coverage']:.2f}  |P|={row['front_len']:.0f}")
res["winners_curse"] = sim_rows
res["bias_worst"] = max(abs(r["rel_bias"]) for r in sim_rows)
res["coverage_worst"] = min(r["coverage"] for r in sim_rows)
checkpoint("R1 winners curse")

# ================================================= build the primary cells ---
cells = collections.defaultdict(list)
for x in csv.DictReader(open(f"{D}/corpus.csv")):
    Q, ch, t = float(x["Q"]), float(x["chance"]), float(x["latency_ms"])
    if t <= 0:
        continue
    e = 1 - (Q - ch) / (1 - ch)
    if 0 < e < 1:
        cells[(x["source"], x["cell"], x["task"])].append((t, e))
for x in csv.DictReader(open(f"{D}/mai_data.csv")):
    t, p = float(x["runtime_ms"]), float(x["psnr"])
    if t > 0:
        cells[("mai", x["cell"], "Super-resolution")].append((t, 10 ** (-p / 10)))


def identify(minfront=6, floor_tol=0.999):
    out = {}
    for k, v in cells.items():
        P = pareto(v)
        if len(P) < minfront:
            continue
        f = fit_knee([p[0] for p in P], [p[1] for p in P])
        if f is None or not (0 < f[2] < 8):
            continue
        if f[0] <= 1e-6 or f[0] >= floor_tol * min(p[1] for p in P):
            continue
        out[k] = dict(P=P, eps_inf=f[0], c=f[1], alpha=f[2], lam=f[2] / 2,
                      recs=v)
    return out


ref = identify()
print(f"\nidentified cells (paper rule): {len(ref)}")

# ===================================== R1b per-cell parametric calibration ---
# The generic grid above shows that bias and coverage depend on how noisy the
# population is. So calibrate each real cell against a simulation matched to
# its own fitted law and its own measured scatter.
print("\nR1b per-cell calibration against a matched simulation")
NCAL = int(os.environ.get("NCAL", 60))
cal_rows = []
for k, d in sorted(ref.items()):
    recs = np.array(d["recs"])
    t_all, e_all = recs[:, 0], recs[:, 1]
    pred = d["eps_inf"] + d["c"] * t_all ** (-d["alpha"])
    sigma = float(np.std(np.log(np.clip(e_all, 1e-9, None) /
                                np.clip(pred, 1e-9, None))))
    n_sys = len(recs)
    lo_t, hi_t = float(t_all.min()), float(t_all.max())
    ah, cov, ntry, zs = [], 0, 0, []
    for _ in range(NCAL):
        t = np.exp(RNG.uniform(math.log(lo_t), math.log(hi_t), n_sys))
        e = np.clip((d["eps_inf"] + d["c"] * t ** (-d["alpha"])) *
                    np.exp(RNG.normal(0, sigma, n_sys)), 1e-6, .999)
        P = pareto(list(zip(t, e)))
        if len(P) < 6:
            continue
        f = fit_knee([p[0] for p in P], [p[1] for p in P],
                     seed=(d["eps_inf"], d["c"], d["alpha"]))
        if f is None or not (0 < f[2] < 8):
            continue
        if f[0] <= 1e-6 or f[0] >= .999 * min(p[1] for p in P):
            continue
        ah.append(f[2])
        ntry += 1
        Pa = np.array(P)
        bl = []
        for _b in range(40):
            idx = RNG.integers(0, len(Pa), len(Pa))
            tt, ee2 = Pa[idx, 0], Pa[idx, 1]
            if len(np.unique(tt)) < 4:
                continue
            g = fit_knee(tt, ee2, seed=(f[0], f[1], f[2]))
            if g and 0 < g[2] < 8:
                bl.append(g[2])
        if len(bl) > 12:
            lo, hi = np.percentile(bl, [2.5, 97.5])
            cov += int(lo <= d["alpha"] <= hi)
            sd = float(np.std(bl, ddof=1))
            if sd > 1e-9:
                zs.append(abs(d["alpha"] - f[2]) / sd)
    if not ah:
        continue
    ah = np.array(ah)
    row = dict(cell=k[1], sigma=sigma, n=n_sys, n_ok=len(ah),
               alpha_fitted=d["alpha"],
               alpha_recovered=float(ah.mean()),
               rel_bias=float(ah.mean() / d["alpha"] - 1),
               coverage=cov / max(ntry, 1),
               inflation=(float(np.percentile(zs, 95)) / 1.96
                          if len(zs) > 10 else None))
    cal_rows.append(row)
    print(f"   {str(k[1])[:40]:40s} sigma={sigma:.2f} "
          f"bias={100*row['rel_bias']:+5.1f}%  coverage={row['coverage']:.2f}")
res["calibration"] = cal_rows
checkpoint("R1b calibration")
if cal_rows:
    b = np.array([r["rel_bias"] for r in cal_rows])
    cv = np.array([r["coverage"] for r in cal_rows])
    lam = np.array([ref[k]["lam"] for k in sorted(ref) if any(
        r["cell"] == k[1] for r in cal_rows)])
    inf_ = [r["inflation"] for r in cal_rows if r["inflation"]]
    res["calibration_median_inflation"] = (float(np.median(inf_))
                                           if inf_ else None)
    res["calibration_median_bias"] = float(np.median(b))
    res["calibration_median_coverage"] = float(np.median(cv))
    corrected = float(np.median([ref[k]["lam"] for k in ref]) /
                      (1 + float(np.median(b))))
    res["bias_corrected_median_lambda"] = corrected
    print(f"   median relative bias {100*np.median(b):+.1f}%, "
          f"median coverage {np.median(cv):.2f} (nominal 0.95)")
    if inf_:
        print(f"   interval inflation needed for nominal 95% coverage: "
              f"x{np.median(inf_):.2f}")
    print(f"   bias-corrected median lambda*: {corrected:.3f} "
          f"(uncorrected {np.median([ref[k]['lam'] for k in ref]):.3f})")

# ============================================== R2 system-level bootstrap ----
print(f"\nR2 system-level bootstrap ({BSYS} replicates per cell)")
sysboot = {}
for k, d in sorted(ref.items()):
    recs = np.array(d["recs"])
    lams = []
    for _ in range(BSYS):
        idx = RNG.integers(0, len(recs), len(recs))
        P = pareto(list(map(tuple, recs[idx])))
        if len(P) < 6:
            continue
        f = fit_knee([p[0] for p in P], [p[1] for p in P],
                     seed=(d["eps_inf"], d["c"], d["alpha"]))
        if f is None or not (0 < f[2] < 8):
            continue
        if f[0] <= 1e-6 or f[0] >= .999 * min(p[1] for p in P):
            continue
        lams.append(f[2] / 2)
    if len(lams) < 30:
        continue
    lams = np.array(lams)
    sysboot[k] = dict(se=float(lams.std(ddof=1)),
                      lo=float(np.percentile(lams, 2.5)),
                      hi=float(np.percentile(lams, 97.5)),
                      n=len(lams))
    print(f"   {str(k[1])[:44]:44s} lam*={d['lam']:.3f} "
          f"[{sysboot[k]['lo']:.3f},{sysboot[k]['hi']:.3f}] "
          f"se={sysboot[k]['se']:.3f}")

keys = [k for k in sorted(ref, key=lambda z: ref[z]["lam"]) if k in sysboot]
mu_s, se_s, tau_s, pi_s = dl_pool([ref[k]["lam"] for k in keys],
                                  [sysboot[k]["se"] ** 2 for k in keys])
print(f"   pooled (system-level SEs): {mu_s:.3f} "
      f"CI [{mu_s-1.96*se_s:.3f},{mu_s+1.96*se_s:.3f}] tau={tau_s:.3f} "
      f"PI [{pi_s[0]:.3f},{pi_s[1]:.3f}]  k={len(keys)}")
res["system_bootstrap"] = dict(
    k=len(keys), pooled=mu_s, se=se_s, tau=tau_s,
    ci=[mu_s - 1.96 * se_s, mu_s + 1.96 * se_s], pi=pi_s,
    median_se=float(np.median([sysboot[k]["se"] for k in keys])),
    cells={str(k): sysboot[k] for k in keys})

# ================================================== R3 clustered pooling -----
print("\nR3 clustered pooling")
clusters = collections.defaultdict(list)
for k in ref:
    fam = k[0]
    if fam == "timm":
        fam = "timm (shared checkpoints)"
    clusters[fam].append(k)
cl_mu, cl_se = [], []
for fam, ks in sorted(clusters.items()):
    lam = np.array([ref[k]["lam"] for k in ks])
    m = float(lam.mean())
    # between-cell SE within the cluster; a single-cell cluster keeps its own SE
    if len(ks) > 1:
        sd = float(lam.std(ddof=1) / math.sqrt(len(ks)))
    else:
        sd = float(sysboot.get(ks[0], dict(se=0.1))["se"])
    cl_mu.append(m)
    cl_se.append(sd)
    print(f"   {fam:28s} k={len(ks):2d} mean lam*={m:.3f} se={sd:.3f}")
mu_c, se_c, tau_c, pi_c = dl_pool(cl_mu, [s ** 2 for s in cl_se])
print(f"   pooled over {len(cl_mu)} clusters: {mu_c:.3f} "
      f"CI [{mu_c-1.96*se_c:.3f},{mu_c+1.96*se_c:.3f}]")
res["clustered"] = dict(n_clusters=len(cl_mu), pooled=mu_c, se=se_c,
                        ci=[mu_c - 1.96 * se_c, mu_c + 1.96 * se_c],
                        cluster_means=cl_mu,
                        labels=sorted(clusters))

# ============================================ R4 identifiability sensitivity -
print("\nR4 identifiability-rule sensitivity")
sens = []
for mf in (5, 6, 8, 10, 12):
    for ft in (0.99, 0.999, 0.9999):
        r = identify(mf, ft)
        if not r:
            continue
        lam = np.array([v["lam"] for v in r.values()])
        sens.append(dict(min_front=mf, floor_tol=ft, k=len(r),
                         median=float(np.median(lam)),
                         mean=float(lam.mean())))
for s_ in sens:
    print(f"   min|P|={s_['min_front']:2d} tol={s_['floor_tol']:.4f}  "
          f"k={s_['k']:2d}  median lam*={s_['median']:.3f}")
med = [s_["median"] for s_ in sens]
print(f"   median lambda* over all {len(sens)} rule settings: "
      f"{min(med):.3f} to {max(med):.3f}")
res["rule_sensitivity"] = dict(settings=sens, lo=min(med), hi=max(med))
checkpoint("R4 identifiability sensitivity")

# ================================================== R5 metric dependence -----
print("\nR5 metric dependence on one task and dataset")
metric_rows = []
for k, d in ref.items():
    if "detection" in k[2].lower() or "Object" in k[2]:
        metric_rows.append(dict(cell=k[1], task=k[2], lam=d["lam"],
                                n_front=len(d["P"])))
if os.path.exists(f"{D}/corpus_extended.csv"):
    ex = collections.defaultdict(list)
    for x in csv.DictReader(open(f"{D}/corpus_extended.csv")):
        if x["task"] != "Object detection":
            continue
        t, Q = float(x["latency_ms"]), float(x["Q"])
        if t > 0 and 0 < 1 - Q < 1:
            ex[(x["cell"], x["metric"])].append((t, 1 - Q))
    for k, v in ex.items():
        P = pareto(v)
        if len(P) < 6:
            continue
        f = fit_knee([p[0] for p in P], [p[1] for p in P])
        if f and 1e-6 < f[0] < .999 * min(p[1] for p in P):
            metric_rows.append(dict(cell=k[0], task="Object detection",
                                    metric=k[1], lam=f[2] / 2, n_front=len(P)))
for r in metric_rows:
    print(f"   lam*={r['lam']:.3f}  |P|={r['n_front']:3d}  "
          f"{r.get('metric', 'mAP50-95')}  {r['cell'][:46]}")
res["metric_dependence"] = metric_rows
if len(metric_rows) >= 2:
    lams = [r["lam"] for r in metric_rows]
    res["metric_spread"] = [min(lams), max(lams)]
    print(f"   spread across metrics on COCO detection: "
          f"{min(lams):.3f} to {max(lams):.3f}")

res["_stages_completed"] = res.get("_stages_completed", []) + ["R5 metric dependence"]
res["_complete"] = True
json.dump(res, open(f"{OUT}/robustness.json", "w"), indent=1, default=float)
print("\nwrote robustness.json (all stages complete)")
