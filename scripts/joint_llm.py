#!/usr/bin/env python3
"""
joint_llm.py -- pooling the language-model evidence properly.

Section 8.4.5 fits each ladder on its own and asks AICc, per ladder, whether the
floor was worth a third parameter.  On a six-point front that question has
almost no power, and the per-ladder verdict is not the question the theory
raises anyway.  The theory says alpha is a property of a frontier: every ladder
measuring the same benchmark should share it, whatever family, GPU or backend
produced the ladder.  That is a constraint, and a constraint bought with one
parameter can pay for a hundred.

Three analyses, all on the 125 identified ladders and their 790 front points.

  A  A joint fit.  Five nested specifications are compared on one likelihood.
     The decisive pair is a saturating law with a single exponent shared by
     every ladder in the study against a pure power law with a free exponent
     for each: 376 parameters against 375.  If the first wins at equal
     complexity, then one global price plus per-ladder floors describes the
     data better than 125 unconstrained slopes, which is the theory's claim
     stated as a model comparison.

  B  Bayes factors.  AICc charges a fixed penalty for eps_inf regardless of how
     much of its range the data exclude.  The Bayesian treatment integrates it
     out under a uniform prior on [0, eps_min] -- the same interval the
     identifiability rule uses -- with c, alpha and the noise scale profiled.
     This is the comparison AICc approximates badly on short fronts.

  C  Prediction.  Fit the price on Qwen1.5 and Qwen2.5, predict Qwen3, and
     score the prediction against the null of predicting the corpus mean.
     Nothing about Qwen3 enters the fit.

Reads   stats_latladders.json
Writes  stats_joint.json
"""
import collections
import json
import math
import os

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

DATA = os.environ.get("DATA", "data")
OUT = os.environ.get("OUT", "results")
RNG = np.random.default_rng(20260824)
NPERM = int(os.environ.get("NPERM", 4000))

S = json.load(open(f"{OUT}/stats_latladders.json"))
lad = []
for c in S["cells"]:
    P = np.asarray(c["points"], float)
    lad.append(dict(label=c["label"], task=c["task"],
                    family=c["label"].split("|")[0].strip(),
                    cls=c["benchmark_class"], x=np.log(P[:, 0]), e=P[:, 1],
                    n=len(P), lam0=c["lam"], eps_min=float(P[:, 1].min())))
tasks = sorted({l["task"] for l in lad})
TI = {t: i for i, t in enumerate(tasks)}
N = sum(l["n"] for l in lad)
print(f"{len(lad)} ladders, {N} front points, {len(tasks)} benchmarks, "
      f"{len({l['family'] for l in lad})} families\n")


def rss_power(l, a):
    """Profile out c at fixed alpha: least squares in log space is exact."""
    u = np.exp(-a * (l["x"] - l["x"].mean()))
    c = float((l["e"] * u).sum() / (u * u).sum())
    return float((((l["e"] - c * u)) ** 2).sum()), c


def rss_sat(l, a, ei=None):
    """Profile c at fixed alpha and floor; optionally profile the floor too."""
    u = np.exp(-a * (l["x"] - l["x"].mean()))
    if ei is None:
        A = np.vstack([np.ones_like(u), u]).T
        coef, *_ = np.linalg.lstsq(A, l["e"], rcond=None)
        ei = min(max(coef[0], 0.0), l["eps_min"] * .999)
    v = l["e"] - ei
    c = float((v * u).sum() / (u * u).sum())
    return float(((v - c * u) ** 2).sum()), ei, c


def ll(rss, n):
    """Concentrated Gaussian log-likelihood, noise scale profiled per ladder."""
    return -0.5 * n * (math.log(2 * math.pi * max(rss, 1e-300) / n) + 1)


def best_alpha(l, sat):
    grid = np.linspace(.02, 4.0, 200)
    r = [(rss_sat(l, a)[0] if sat else rss_power(l, a)[0]) for a in grid]
    a0 = grid[int(np.argmin(r))]
    f = lambda z: (rss_sat(l, abs(z[0]))[0] if sat else rss_power(l, abs(z[0]))[0])
    o = minimize(f, [a0], method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-12))
    return abs(float(o.x[0]))


def fit_shared(group, sat):
    """One alpha for a group of ladders; everything else profiled per ladder."""
    def obj(z):
        a = abs(z[0])
        return -sum(ll((rss_sat(l, a)[0] if sat else rss_power(l, a)[0]), l["n"])
                    for l in group)
    grid = np.linspace(.05, 3.0, 120)
    a0 = grid[int(np.argmin([obj([a]) for a in grid]))]
    o = minimize(obj, [a0], method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-10))
    return abs(float(o.x[0])), -float(o.fun)


def aicc(logL, k, n):
    return -2 * logL + 2 * k + (2 * k * (k + 1) / (n - k - 1)
                                if n - k - 1 > 0 else float("inf"))


# =============================================================== A  joint ===
print("A  joint fit: five specifications on one likelihood\n")
models = {}

L = sum(ll(rss_power(l, best_alpha(l, False))[0], l["n"]) for l in lad)
models["power, free alpha per ladder"] = (L, 3 * len(lad))

L = sum(ll(rss_sat(l, best_alpha(l, True))[0], l["n"]) for l in lad)
models["saturating, free alpha per ladder"] = (L, 4 * len(lad))

bytask = collections.defaultdict(list)
for l in lad:
    bytask[l["task"]].append(l)
for name, sat, per in (("power, alpha shared by benchmark", False, 2),
                       ("saturating, alpha shared by benchmark", True, 3)):
    tot, alphas = 0.0, {}
    for t, g in sorted(bytask.items()):
        a, lg = fit_shared(g, sat)
        alphas[t] = a
        tot += lg
    models[name] = (tot, per * len(lad) + len(tasks))
    if sat:
        shared_by_task = alphas

a_glob, L = fit_shared(lad, True)
models["saturating, one alpha for the whole study"] = (L, 3 * len(lad) + 1)

ref = models["power, free alpha per ladder"]
rows = []
for name, (logL, k) in models.items():
    A = aicc(logL, k, N)
    rows.append((name, logL, k, A))
best = min(r[3] for r in rows)
print(f"  {'specification':40s} {'k':>5} {'logL':>10} {'dAICc':>9}")
for name, logL, k, A in rows:
    print(f"  {name:40s} {k:5d} {logL:10.1f} {A - best:+9.1f}")
mg = models["saturating, one alpha for the whole study"]
mp = models["power, free alpha per ladder"]
print(f"\n  the decisive pair, at essentially equal complexity "
      f"({mg[1]} against {mp[1]} parameters):")
print(f"  one global exponent plus per-ladder floors beats 125 free exponents "
      f"by dAICc = {aicc(*mp[::1][:2], N) if False else aicc(mp[0], mp[1], N) - aicc(mg[0], mg[1], N):+.1f}")
print(f"  the shared exponent is alpha = {a_glob:.3f}, "
      f"lambda* = {a_glob / 2:.3f}")
lr = 2 * (models["saturating, alpha shared by benchmark"][0]
          - models["power, alpha shared by benchmark"][0])
df = len(lad)
print(f"  likelihood ratio for a floor, alpha shared by benchmark: "
      f"{lr:.1f} on {df} df, p = {1 - chi2.cdf(lr, df):.2e}")

# ============================================================ B  Bayes ======
print("\nB  Bayes factors with the floor integrated out\n")


def logbf(l, ngrid=160):
    a_p = best_alpha(l, False)
    l0 = ll(rss_power(l, a_p)[0], l["n"])
    grid = np.linspace(1e-9, l["eps_min"] * .999, ngrid)
    vals = []
    for ei in grid:
        a = best_alpha_at_floor(l, ei)
        vals.append(ll(rss_sat(l, a, ei)[0], l["n"]))
    v = np.array(vals)
    m = v.max()
    marg = m + math.log(np.trapezoid(np.exp(v - m), grid)
                        / (l["eps_min"] * .999))
    return marg - l0


def best_alpha_at_floor(l, ei):
    grid = np.linspace(.02, 4.0, 60)
    r = [rss_sat(l, a, ei)[0] for a in grid]
    return float(grid[int(np.argmin(r))])


bfs = {l["label"]: logbf(l) for l in lad}
units = collections.defaultdict(list)
for l in lad:
    units[(l["family"], l["task"])].append(bfs[l["label"]])
ub = {k: float(np.mean(v)) for k, v in units.items()}
tot = float(sum(ub.values()))
pos = sum(1 for v in ub.values() if v > 0)
print(f"  mean log Bayes factor per ladder {np.mean(list(bfs.values())):+.2f}")
print(f"  summed over the {len(ub)} family-by-benchmark units: "
      f"{tot:+.1f} log units, favourable in {pos}/{len(ub)}")
for cls in ("retired", "headroom"):
    v = [ub[k] for k in ub
         if any(l["cls"] == cls for l in lad
                if (l["family"], l["task"]) == k)]
    if v:
        print(f"  {cls:9s}: summed {sum(v):+.1f}, "
              f"favourable in {sum(1 for x in v if x > 0)}/{len(v)}")

# ======================================================== C  prediction =====
print("\nC  out-of-sample: fit on Qwen1.5 and Qwen2.5, predict Qwen3\n")
train = [l for l in lad if l["family"] in ("qwen1.5", "qwen2.5")]
test = [l for l in lad if l["family"] == "qwen3"]
tt = collections.defaultdict(list)
for l in train:
    tt[l["task"]].append(l)
pred = {t: fit_shared(g, True)[0] for t, g in sorted(tt.items())}
grand = float(np.mean(list(pred.values())))
have = [l for l in test if l["task"] in pred]
if have:
    obs = np.array([best_alpha(l, True) for l in have])
    pr = np.array([pred[l["task"]] for l in have])
    rmse = float(np.sqrt(np.mean((obs - pr) ** 2)))
    null = float(np.sqrt(np.mean((obs - grand) ** 2)))
    perm = []
    keys = list(pred)
    for _ in range(NPERM):
        sh = dict(zip(keys, RNG.permutation([pred[k] for k in keys])))
        perm.append(np.sqrt(np.mean((obs - np.array([sh[l["task"]]
                                                     for l in have])) ** 2)))
    p = float(np.mean(np.array(perm) <= rmse))
    print(f"  {len(have)} held-out Qwen3 ladders on "
          f"{len({l['task'] for l in have})} benchmarks")
    print(f"  RMSE of the benchmark-specific prediction {rmse:.3f}")
    print(f"  RMSE of predicting the corpus mean         {null:.3f}"
          f"   ({100 * (1 - rmse / null):+.0f}% )")
    print(f"  permutation p (benchmark labels shuffled)  {p:.4f}")
else:
    rmse = null = p = None

json.dump(dict(
    n_ladders=len(lad), n_points=N, n_tasks=len(tasks),
    joint={n: dict(logL=v[0], k=v[1], aicc=aicc(v[0], v[1], N))
           for n, v in models.items()},
    decisive_daicc=float(aicc(mp[0], mp[1], N) - aicc(mg[0], mg[1], N)),
    alpha_global=a_glob, lambda_global=a_glob / 2,
    alpha_by_task=shared_by_task,
    lrt=dict(stat=float(lr), df=df, p=float(1 - chi2.cdf(lr, df))),
    bayes=dict(mean_log_bf=float(np.mean(list(bfs.values()))),
               unit_total=tot, unit_favourable=pos, n_units=len(ub),
               per_unit={f"{a} | {b}": v for (a, b), v in ub.items()}),
    prediction=dict(n=len(have), rmse=rmse, null_rmse=null, perm_p=p,
                    predicted={k: v for k, v in pred.items()}),
), open(f"{OUT}/stats_joint.json", "w"), indent=1, default=float)
print(f"\nwrote {OUT}/stats_joint.json")
