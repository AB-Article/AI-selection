#!/usr/bin/env python3
"""
reconcile.py  --  make the corrections of robustness.py consistent with every
number that depends on them, and re-examine the transfer claim honestly.

C1  Transfer. Inverse-variance pooling gives the ten out-of-domain cells almost
    no weight, so "the pooled price is unchanged when they are added" is close
    to arithmetically guaranteed. Measure the weight share, and pool the
    out-of-domain cells on their own so that the transfer claim rests on a
    comparison of two independent estimates rather than on a weighted average
    dominated by one of them.

C2  Coverage. robustness.py finds that the nominal 95% front-level bootstrap
    intervals cover at 0.66 and need widening by x1.36. Those intervals are the
    within-cell variances of the random-effects pooling, so every pooled
    estimate, tau, Q, I^2 and prediction interval must be recomputed with the
    inflated standard errors.

C3  Bias. The calibration measures the bias of alpha-hat using the fitted alpha
    as the simulated truth, which is itself biased. Iterate to a fixed point:
    simulate at alpha_true = alpha_hat / (1 + b), remeasure b, repeat.

Writes reconciled.json
"""
import collections
import csv
import json
import math
import os
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2, t as tdist

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260824)

_D = os.environ.get("DATA", "data")
_OUT = os.environ.get("OUT", "results")
D, OUT = _D, _OUT
os.makedirs(OUT, exist_ok=True)

NCAL = int(os.environ.get("NCAL", 60))
NITER = int(os.environ.get("NITER", 4))


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
              if seed else [[min(ee) * s, 0., .5] for s in (0., .3, .6, .9)])
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


def dl(y, s2):
    """DerSimonian-Laird with full diagnostics."""
    y, s2 = np.asarray(y, float), np.maximum(np.asarray(s2, float), 1e-12)
    k = len(y)
    if k < 2:
        se = math.sqrt(s2[0])
        return dict(k=k, mu=float(y[0]), se=se, tau=0.0, Q=0.0, Q_p=1.0,
                    I2=0.0, ci=[y[0] - 1.96 * se, y[0] + 1.96 * se],
                    pi=[float(y[0])] * 2, weights=[1.0])
    w = 1 / s2
    mufe = (w * y).sum() / w.sum()
    Q = float((w * (y - mufe) ** 2).sum())
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C)
    wr = 1 / (s2 + tau2)
    mu = float((wr * y).sum() / wr.sum())
    se = float(math.sqrt(1 / wr.sum()))
    tc = float(tdist.ppf(0.975, max(k - 2, 1)))
    return dict(k=k, mu=mu, se=se, tau=math.sqrt(tau2), Q=Q,
                Q_p=float(1 - chi2.cdf(Q, k - 1)),
                I2=max(0.0, 100 * (Q - (k - 1)) / Q) if Q > 0 else 0.0,
                ci=[mu - 1.96 * se, mu + 1.96 * se],
                pi=[mu - tc * math.sqrt(tau2 + se ** 2),
                    mu + tc * math.sqrt(tau2 + se ** 2)],
                weights=list(wr / wr.sum()))


res = {}
INFL = 1.36   # interval inflation from robustness.py

# ------------------------------------------------------------------ load ----
M = json.load(open(f"{OUT}/stats_extended.json"))
cells = M["cells"]
lam = np.array([c["lam"] for c in cells])
se = np.array([c["se"] for c in cells])
grp = np.array([c["group"] for c in cells])
task = np.array([c["task"] for c in cells])
prim = grp == "Primary vision inference"

# ======================================================= C2 propagate SEs ----
print("C2  propagating the coverage correction (SE x%.2f)" % INFL)
sets = {"primary 12": prim, "all 22": np.ones(len(lam), bool),
        "out-of-domain 10": ~prim}
res["pooling"] = {}
for name, m in sets.items():
    for tag, f in (("uncorrected", 1.0), ("corrected", INFL)):
        p = dl(lam[m], (se[m] * f) ** 2)
        res["pooling"][f"{name} / {tag}"] = p
        print(f"   {name:18s} {tag:12s} k={p['k']:2d} pooled {p['mu']:.4f} "
              f"CI [{p['ci'][0]:.3f},{p['ci'][1]:.3f}] tau={p['tau']:.4f} "
              f"I2={p['I2']:3.0f}%  Q_p={p['Q_p']:.3f} "
              f"PI [{p['pi'][0]:.3f},{p['pi'][1]:.3f}]")

# ======================================================== C1 transfer ---------
print("\nC1  is the transfer claim real?")
p_all = dl(lam, (se * INFL) ** 2)
w = np.array(p_all["weights"])
res["weight_share_primary"] = float(w[prim].sum())
res["weight_share_new"] = float(w[~prim].sum())
print(f"   weight share of the 12 primary cells : {100*w[prim].sum():.1f}%")
print(f"   weight share of the 10 new cells     : {100*w[~prim].sum():.1f}%")

p_new = dl(lam[~prim], (se[~prim] * INFL) ** 2)
p_pri = dl(lam[prim], (se[prim] * INFL) ** 2)
diff = p_new["mu"] - p_pri["mu"]
sed = math.sqrt(p_new["se"] ** 2 + p_pri["se"] ** 2)
z = diff / sed
res["independent_comparison"] = dict(
    primary=p_pri["mu"], primary_ci=p_pri["ci"],
    new=p_new["mu"], new_ci=p_new["ci"],
    diff=diff, se_diff=sed, z=z,
    p_two_sided=float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))))
print(f"   primary alone      : {p_pri['mu']:.3f} CI [{p_pri['ci'][0]:.3f},{p_pri['ci'][1]:.3f}]")
print(f"   out-of-domain alone: {p_new['mu']:.3f} CI [{p_new['ci'][0]:.3f},{p_new['ci'][1]:.3f}]")
print(f"   difference {diff:+.3f} +- {sed:.3f}  z={z:.2f}  "
      f"p={res['independent_comparison']['p_two_sided']:.3f}")

for name, m in (("primary", prim), ("out-of-domain", ~prim)):
    print(f"   {name:14s} unweighted mean {lam[m].mean():.3f}  "
          f"median {np.median(lam[m]):.3f}  range "
          f"[{lam[m].min():.3f},{lam[m].max():.3f}]")
res["unweighted"] = {n: dict(mean=float(lam[m].mean()),
                             median=float(np.median(lam[m])),
                             lo=float(lam[m].min()), hi=float(lam[m].max()))
                     for n, m in (("primary", prim), ("new", ~prim))}

# ============================================ C3 fixed-point bias iteration --
cellrecs = collections.defaultdict(list)
for x in csv.DictReader(open(f"{D}/corpus.csv")):
    Q, ch, t = float(x["Q"]), float(x["chance"]), float(x["latency_ms"])
    if t <= 0:
        continue
    e = 1 - (Q - ch) / (1 - ch)
    if 0 < e < 1:
        cellrecs[(x["source"], x["cell"])].append((t, e))

fitted = {}
for k, v in cellrecs.items():
    P = pareto(v)
    if len(P) < 6:
        continue
    f = fit_knee([p[0] for p in P], [p[1] for p in P])
    if f and 1e-6 < f[0] < .999 * min(p[1] for p in P) and 0 < f[2] < 8:
        fitted[k] = dict(P=P, recs=np.array(v), eps_inf=f[0], c=f[1], alpha=f[2])
print(f"\nC3  fixed-point bias iteration over {len(fitted)} cells")


def measure_bias(scale):
    """Simulate at alpha_true = alpha_hat/scale, return median relative bias."""
    out = []
    for k, d in fitted.items():
        recs = d["recs"]
        t_all = recs[:, 0]
        pred = d["eps_inf"] + d["c"] * t_all ** (-d["alpha"])
        sigma = float(np.std(np.log(np.clip(recs[:, 1], 1e-9, None) /
                                    np.clip(pred, 1e-9, None))))
        a_true = d["alpha"] / scale
        lo_t, hi_t = float(t_all.min()), float(t_all.max())
        ah = []
        for _ in range(NCAL):
            t = np.exp(RNG.uniform(math.log(lo_t), math.log(hi_t), len(recs)))
            e = np.clip((d["eps_inf"] + d["c"] * t ** (-a_true)) *
                        np.exp(RNG.normal(0, sigma, len(recs))), 1e-6, .999)
            P = pareto(list(zip(t, e)))
            if len(P) < 6:
                continue
            f = fit_knee([p[0] for p in P], [p[1] for p in P],
                         seed=(d["eps_inf"], d["c"], a_true))
            if f is None or not (0 < f[2] < 8):
                continue
            if f[0] <= 1e-6 or f[0] >= .999 * min(p[1] for p in P):
                continue
            ah.append(f[2])
        if ah:
            out.append(float(np.mean(ah)) / a_true - 1)
    return float(np.median(out)) if out else 0.0


scale, hist = 1.0, []
for it in range(NITER):
    b = measure_bias(scale)
    hist.append(dict(iteration=it, scale=scale, bias=b))
    print(f"   iter {it}: simulate at alpha_hat/{scale:.4f} -> bias {100*b:+.2f}%")
    new_scale = 1 + b
    if abs(new_scale - scale) < 0.005:
        scale = new_scale
        break
    scale = new_scale
res["bias_iterations"] = hist
res["bias_fixed_point"] = scale - 1
med_lam = float(np.median(lam[prim]))
res["lambda_corrected_fixed_point"] = med_lam / scale
print(f"   fixed point: bias {100*(scale-1):+.2f}%, "
      f"corrected median lambda* = {med_lam/scale:.3f} (uncorrected {med_lam:.3f})")

json.dump(res, open(f"{OUT}/reconciled.json", "w"), indent=1)
print("\nwrote reconciled.json")
