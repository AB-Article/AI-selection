#!/usr/bin/env python3
"""
retired_stats.py -- the inferential half of the positive control.

analysis_retired.py fits every ladder.  This script asks the four questions the
fits were collected to answer, with inference that respects the clustering of
ladders inside model suites.

  Q1  Is the evidence for a floor ordered by how far the front has descended?
      Predictor: S_max = -ln eps_min, the resolved information of the best rung.
      It is an endpoint of the data, fixed before the saturating law is fitted.
  Q2  Does the ordering survive clustering?  Ladders that share a suite, a size
      or a checkpoint series are not independent, so every correlation is
      re-estimated by a cluster bootstrap over checkpoint series.
  Q3  Where a human ceiling is published, does the fitted floor respect it?
      Under a null in which the fit carries no information about the ceiling,
      eps_inf is uniform on [0, eps_min) and the chance of landing below the
      ceiling is A = eps_c / eps_min.  Summing those probabilities gives a
      Poisson-binomial reference for the observed number of violations.
  Q4  Is the degeneracy a property of the domain or of the range?  Censoring
      the top rungs of a ladder -- language-model or ImageNet -- should move it
      onto the same identification curve.

Reads   table_retired.csv, table_retired_tokens.csv, stats_retired.json,
        corpus.csv, table_ladders.csv (optional, for the earlier LLM ladders)
Writes  stats_retired_final.json, table_retired_summary.csv, table_censoring.csv
"""
import collections
import csv
import json
import math
import os
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, t as tdist, chi2

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260824)

DATA = os.environ.get("DATA", "data")
OUT = os.environ.get("OUT", "results")
MINFRONT = 6
NBOOT = int(os.environ.get("CLUSTER_BOOT", 2000))

# Published human accuracies, in the benchmark's own units.  Only benchmarks
# whose source paper states a human number are listed; SciQ, ARC-Easy and
# ARC-Challenge state none, and LAMBADA was constructed so that its retained
# passages are solved by human subjects, which puts its ceiling at essentially
# one and makes the test below vacuous rather than informative.  Those four are
# therefore excluded from Q3 and enter only through Q1, Q2 and Q4.
CEILING = {"PIQA": 0.950,        # Bisk et al. 2020
           "Winogrande": 0.940,  # Sakaguchi et al. 2020
           "HellaSwag": 0.956}   # Zellers et al. 2019
CHANCE = {"SciQ": .25, "PIQA": .50, "ARC-Easy": .25, "LAMBADA": .0,
          "Winogrande": .50, "HellaSwag": .25, "ARC-Challenge": .25,
          "LogiQA": .25, "WSC": .50}


def eps_ceiling(task):
    h = CEILING.get(task)
    return None if h is None else max(1e-6, 1 - (h - CHANCE[task]) / (1 - CHANCE[task]))


# --------------------------------------------------------------- machinery --
def pareto(pts):
    out, best = [], np.inf
    for t, e in sorted(pts):
        if e < best - 1e-15:
            out.append((t, e))
            best = e
    return out


def fit_knee(t, e):
    lr, ee = np.log(np.asarray(t, float)), np.asarray(e, float)
    f = lambda lr, ei, lc, a: ei + np.exp(lc) * np.exp(-a * lr)
    best = None
    for s in (0., .3, .6, .9):
        try:
            p, _ = curve_fit(f, lr, ee, p0=[min(ee) * s, 0., .5], maxfev=400000,
                             bounds=([0, -60, .01], [min(ee) * .999, 60, 8]))
        except Exception:
            continue
        r = ee - f(lr, *p)
        r2 = 1 - (r ** 2).sum() / max(((ee - ee.mean()) ** 2).sum(), 1e-300)
        if best is None or r2 > best[3]:
            best = (p[0], math.exp(p[1]), p[2], r2)
    return best


def fit_power(t, e):
    lr, ee = np.log(np.asarray(t, float)), np.asarray(e, float)
    f = lambda lr, lc, a: np.exp(lc) * np.exp(-a * lr)
    try:
        p, _ = curve_fit(f, lr, ee, p0=[0., .5], maxfev=400000,
                         bounds=([-60, .0], [60, 8]))
    except Exception:
        return None
    return math.exp(p[0]), p[1], float(((ee - f(lr, *p)) ** 2).sum())


def aicc(rss, n, k):
    return np.nan if n - k - 1 <= 0 else \
        n * math.log(max(rss, 1e-300) / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def dl(y, s2):
    y = np.asarray(y, float)
    s2 = np.maximum(np.asarray(s2, float), 1e-12)
    k = len(y)
    if k == 0:
        return None
    if k == 1:
        se = math.sqrt(s2[0])
        return dict(k=1, mu=float(y[0]), se=se, tau=.0, I2=.0,
                    ci=[float(y[0] - 1.96 * se), float(y[0] + 1.96 * se)],
                    pi=[float(y[0]), float(y[0])])
    w = 1 / s2
    mufe = (w * y).sum() / w.sum()
    Q = float((w * (y - mufe) ** 2).sum())
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0., (Q - (k - 1)) / C)
    wr = 1 / (s2 + tau2)
    mu = float((wr * y).sum() / wr.sum())
    se = float(math.sqrt(1 / wr.sum()))
    tc = float(tdist.ppf(.975, max(k - 2, 1)))
    return dict(k=k, mu=mu, se=se, tau=math.sqrt(tau2),
                I2=max(0., 100 * (Q - (k - 1)) / Q) if Q > 0 else .0,
                Q_p=float(1 - chi2.cdf(Q, k - 1)),
                ci=[mu - 1.96 * se, mu + 1.96 * se],
                pi=[mu - tc * math.sqrt(tau2 + se ** 2),
                    mu + tc * math.sqrt(tau2 + se ** 2)])


def cluster_boot_rho(x, y, cluster, n=NBOOT):
    """Spearman with a bootstrap over clusters rather than over ladders."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    groups = collections.defaultdict(list)
    for i, c in enumerate(cluster):
        groups[c].append(i)
    keys = list(groups)
    out = []
    for _ in range(n):
        idx = []
        for k in RNG.choice(len(keys), len(keys), replace=True):
            idx += groups[keys[k]]
        if len(set(y[idx])) < 2 or len(set(x[idx])) < 2:
            continue
        out.append(spearmanr(x[idx], y[idx]).statistic)
    out = np.array([o for o in out if np.isfinite(o)])
    return (float(spearmanr(x, y).statistic),
            float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)),
            float(np.mean(out <= 0)))


# ------------------------------------------------------------------ inputs --
rows = []
for fn in ("table_retired.csv", "table_retired_tokens.csv"):
    path = os.path.join(OUT, fn)
    if not os.path.exists(path):
        continue
    for r in csv.DictReader(open(path)):
        if int(r["n_front"] or 0) < MINFRONT:
            continue
        for k in ("eps_min", "eps_inf", "alpha", "lambda_star", "R2", "R2_power",
                  "S_range", "x_range", "d_aicc"):
            r[k] = float(r[k]) if r[k] not in ("", None) else None
        r["n_front"] = int(r["n_front"])
        r["identified"] = r["status"] == "identified"
        r["S_max"] = -math.log(r["eps_min"])
        r["series"] = f"{r['suite']}|{r.get('model', '')}|{r['shots']}"
        rows.append(r)

stats = json.load(open(os.path.join(OUT, "stats_retired.json")))
cells = {c["label"]: c for c in stats["cells"]}

# =============================================================== Q1 ordering
by_task = {}
for t, v in sorted(collections.Counter(r["task"] for r in rows).items()):
    sub = [r for r in rows if r["task"] == t]
    da = [r["d_aicc"] for r in sub if r["d_aicc"] is not None]
    by_task[t] = dict(
        n=len(sub), identified=sum(r["identified"] for r in sub),
        eps_min=float(np.median([r["eps_min"] for r in sub])),
        S_max=float(np.median([r["S_max"] for r in sub])),
        d_aicc=float(np.median(da)) if da else None,
        saturating=int(sum(1 for x in da if x > 0)), n_aicc=len(da),
        R2=float(np.median([r["R2"] for r in sub if r["R2"] is not None]))
        if any(r["R2"] is not None for r in sub) else None,
        ceiling=eps_ceiling(t))

print("Q1  evidence for a floor, ordered by how far the front descends\n")
print(f"  {'benchmark':14s} {'n':>4} {'ident':>6} {'eps_min':>8} {'S_max':>6} "
      f"{'dAICc':>7} {'saturating':>11}")
for t, d in sorted(by_task.items(), key=lambda z: z[1]["eps_min"]):
    print(f"  {t:14s} {d['n']:4d} {d['identified']:6d} {d['eps_min']:8.3f} "
          f"{d['S_max']:6.2f} {d['d_aicc']:+7.2f} "
          f"{d['saturating']:5d}/{d['n_aicc']:<5d}")

# =============================================================== Q2 clustering
have = [r for r in rows if r["d_aicc"] is not None]
r1 = cluster_boot_rho([r["S_max"] for r in have], [r["d_aicc"] for r in have],
                      [r["series"] for r in have])
r2 = cluster_boot_rho([r["S_max"] for r in rows],
                      [float(r["identified"]) for r in rows],
                      [r["series"] for r in rows])
r3 = cluster_boot_rho([r["n_front"] for r in have], [r["d_aicc"] for r in have],
                      [r["series"] for r in have])
r4 = cluster_boot_rho([r["S_range"] or 0 for r in have],
                      [r["d_aicc"] for r in have], [r["series"] for r in have])
tasks = sorted(by_task, key=lambda t: by_task[t]["eps_min"])
rho_task = spearmanr([by_task[t]["eps_min"] for t in tasks],
                     [by_task[t]["d_aicc"] for t in tasks])

print(f"\nQ2  cluster bootstrap over {len({r['series'] for r in rows})} "
      f"checkpoint series, {NBOOT} resamples\n")
for name, r in (("S_max vs dAICc", r1), ("S_max vs identified", r2),
                ("front length vs dAICc", r3), ("front span vs dAICc", r4)):
    print(f"  rho({name:22s}) = {r[0]:+.2f}  [{r[1]:+.2f},{r[2]:+.2f}]  "
          f"P(rho<=0) = {r[3]:.4f}")
print(f"  across the {len(tasks)} benchmarks, rho(median eps_min, median dAICc) "
      f"= {rho_task.statistic:+.2f} (p = {rho_task.pvalue:.3f})")

# =============================================================== Q3 ceiling
anchor = []
for r in rows:
    if not r["identified"]:
        continue
    ec = eps_ceiling(r["task"])
    if ec is None:
        continue
    c = cells.get(r["label"], {})
    anchor.append(dict(label=r["label"], task=r["task"], resource=r["resource"],
                       eps_inf=r["eps_inf"], eps_min=r["eps_min"], eps_c=ec,
                       A=ec / r["eps_min"], respects=r["eps_inf"] >= ec,
                       lo=c.get("floor_lo"), hi=c.get("floor_hi"),
                       covers=bool(c.get("floor_lo") is not None
                                   and c["floor_lo"] <= ec <= c["floor_hi"])))
viol = [a for a in anchor if not a["respects"]]
expA = sum(a["A"] for a in anchor)
# Poisson-binomial tail: P(V <= observed) under eps_inf ~ U(0, eps_min)
if anchor:
    pmf = np.array([1.0])
    for a in anchor:
        pmf = np.convolve(pmf, [1 - a["A"], a["A"]])
    p_left = float(pmf[:len(viol) + 1].sum())
    print(f"\nQ3  the fitted floor against the published ceiling\n")
    print(f"  {len(anchor)} identified ladders on benchmarks with a published "
          f"human ceiling")
    print(f"  floors respecting eps_inf >= eps_ceiling: "
          f"{len(anchor) - len(viol)}/{len(anchor)}")
    print(f"  expected violations if the floor were uninformative: "
          f"{expA:.1f}; observed {len(viol)}; P(V <= obs) = {p_left:.2e}")
    print(f"  bootstrap interval of the floor covers the published ceiling in "
          f"{sum(a['covers'] for a in anchor)}/{len(anchor)}")
else:
    p_left = None

# =============================================================== Q4 censoring
print("\nQ4  censoring: identification as a function of the range that is left\n")


def censor(front, name, kind):
    out = []
    for drop in range(0, len(front) - MINFRONT + 1):
        sub = front[:len(front) - drop]
        t = np.array([p[0] for p in sub], float)
        e = np.array([p[1] for p in sub], float)
        f = fit_knee(t, e)
        if f is None:
            continue
        ei, c, a, r2 = f
        pw = fit_power(t, e)
        rk = float(((e - (ei + c * t ** (-a))) ** 2).sum())
        d = (float(aicc(pw[2], len(t), 2) - aicc(rk, len(t), 3))
             if pw is not None else float("nan"))
        out.append(dict(name=name, kind=kind, dropped=drop, n=len(sub),
                        S_max=round(-math.log(e.min()), 4),
                        eps_inf=round(ei, 6), lam=round(a / 2, 4),
                        R2=round(r2, 4), d_aicc=round(d, 2),
                        identified=bool(not (ei <= 1e-6 or ei >= .999 * e.min()))))
    return out


curves = []
deep = [r for r in rows if r["identified"] and r["n_front"] >= 9
        and r["task"] in ("SciQ", "LAMBADA", "PIQA", "ARC-Easy")]
for r in deep:
    c = cells.get(r["label"])
    if c:
        curves += censor([tuple(p) for p in c["points"]], r["label"], "LLM ladder")

prim = collections.defaultdict(list)
path = os.path.join(DATA, "corpus.csv")
if os.path.exists(path):
    for r in csv.DictReader(open(path)):
        try:
            Q, ch, t = float(r["Q"]), float(r["chance"]), float(r["latency_ms"])
        except (TypeError, ValueError):
            continue
        if t <= 0:
            continue
        e = 1 - (Q - ch) / (1 - ch)
        if 0 < e < 1:
            prim[r["cell"]].append((t, e))
nprim = 0
for cell, pts in sorted(prim.items()):
    front = pareto(pts)
    if len(front) < MINFRONT + 3:
        continue
    e = np.array([p[1] for p in front])
    f = fit_knee(np.array([p[0] for p in front]), e)
    if f is None or f[0] <= 1e-6 or f[0] >= .999 * e.min():
        continue
    nprim += 1
    curves += censor(front, cell, "primary cell")

for kind in ("LLM ladder", "primary cell"):
    sub = [c for c in curves if c["kind"] == kind]
    if not sub:
        continue
    names = {c["name"] for c in sub}
    lost = 0
    for n in names:
        s = sorted([c for c in sub if c["name"] == n], key=lambda z: z["dropped"])
        if s[0]["identified"] and not s[-1]["identified"]:
            lost += 1
    print(f"  {kind:13s}: {len(names):3d} curves, {len(sub):4d} refits; "
          f"identification lost under censoring in {lost}/{len(names)}")

# pooled identification rate in bins of remaining resolved information
edges = [0, .3, .5, .7, 1.0, 1.5, 10]
bins = []
for i in range(len(edges) - 1):
    sel = [c for c in curves if edges[i] <= c["S_max"] < edges[i + 1]]
    if not sel:
        continue
    row = dict(lo=edges[i], hi=edges[i + 1])
    for kind in ("LLM ladder", "primary cell"):
        s = [c for c in sel if c["kind"] == kind]
        row[kind] = (float(np.mean([c["identified"] for c in s])), len(s)) if s else None
    bins.append(row)
print(f"\n  {'S_max bin':16s} {'LLM ladders':>20s} {'primary cells':>20s}")
for b in bins:
    f = lambda z: f"{z[0]:.0%} (n={z[1]})" if z else "-"
    print(f"  [{b['lo']:.1f},{b['hi']:.1f})".ljust(18)
          + f"{f(b['LLM ladder']):>20s} {f(b['primary cell']):>20s}")
both = [b for b in bins if b["LLM ladder"] and b["primary cell"]
        and b["LLM ladder"][1] >= 10 and b["primary cell"][1] >= 10]
gap = (float(np.median([abs(b["LLM ladder"][0] - b["primary cell"][0])
                        for b in both])) if both else None)
if gap is not None:
    print(f"  median absolute difference between the two curves, "
          f"over bins with n>=10 on both sides: {gap:.2f}")

# =============================================================== pooling ====
sat = [r for r in rows if r["identified"] and r["task"] in ("SciQ", "LAMBADA")]
recs = [(r, cells[r["label"]]) for r in sat if r["label"] in cells
        and cells[r["label"]].get("se")]
pool = dl([c["lam"] for _, c in recs], [c["se"] ** 2 for _, c in recs])
stage1, groups = {}, collections.defaultdict(list)
for r, c in recs:
    groups[(r["task"], r["suite"])].append(c)
for k, v in sorted(groups.items()):
    stage1["%s | %s" % k] = dl([c["lam"] for c in v], [c["se"] ** 2 for c in v])
stage2 = dl([p["mu"] for p in stage1.values()],
            [p["se"] ** 2 for p in stage1.values()])
bytask = {}
for t in ("SciQ", "LAMBADA"):
    v = [c for r, c in recs if r["task"] == t]
    if v:
        bytask[t] = dl([c["lam"] for c in v], [c["se"] ** 2 for c in v])

print("\nPooled canonical price on the two saturated benchmarks\n")
print(f"  ladder as unit  k={pool['k']:3d}  lambda* {pool['mu']:.3f} "
      f"[{pool['ci'][0]:.3f},{pool['ci'][1]:.3f}]  tau={pool['tau']:.3f} "
      f"I2={pool['I2']:.0f}%")
print(f"  two-stage       k={stage2['k']:3d}  lambda* {stage2['mu']:.3f} "
      f"[{stage2['ci'][0]:.3f},{stage2['ci'][1]:.3f}]")
for t, p in bytask.items():
    print(f"    {t:10s} k={p['k']:3d}  {p['mu']:.3f} "
          f"[{p['ci'][0]:.3f},{p['ci'][1]:.3f}]  I2={p['I2']:.0f}%")
below = sum(1 for _, c in recs if c.get("hi", 1) < .5)
print(f"  bootstrap interval entirely below 1/2 in {below}/{len(recs)}")

# ------------------------------------------------------------------ outputs --
with open(os.path.join(OUT, "table_retired_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["benchmark", "n_ladders", "n_identified", "median_eps_min",
                "median_S_max", "median_dAICc", "n_saturating_preferred",
                "n_with_aicc", "median_R2", "eps_ceiling"])
    for t, d in sorted(by_task.items(), key=lambda z: z[1]["eps_min"]):
        w.writerow([t, d["n"], d["identified"], round(d["eps_min"], 4),
                    round(d["S_max"], 3),
                    round(d["d_aicc"], 2) if d["d_aicc"] is not None else "",
                    d["saturating"], d["n_aicc"],
                    round(d["R2"], 4) if d["R2"] else "",
                    round(d["ceiling"], 4) if d["ceiling"] else ""])

with open(os.path.join(OUT, "table_censoring.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["name", "kind", "dropped", "n", "S_max",
                                       "eps_inf", "lam", "R2", "d_aicc",
                                       "identified"])
    w.writeheader()
    for c in curves:
        w.writerow(c)

json.dump(dict(
    n_ladders=len(rows), n_series=len({r["series"] for r in rows}),
    by_task=by_task,
    rho=dict(S_max_daicc=r1, S_max_identified=r2, front_daicc=r3, span_daicc=r4,
             across_benchmarks=[float(rho_task.statistic), float(rho_task.pvalue)]),
    anchor=dict(n=len(anchor), violations=len(viol), expected=expA,
                p_left=p_left, rows=anchor,
                covers=int(sum(a["covers"] for a in anchor))),
    censoring=dict(n_llm=len({c["name"] for c in curves
                              if c["kind"] == "LLM ladder"}),
                   n_primary=nprim, bins=bins, median_gap=gap),
    pooled=dict(ladder=pool, two_stage=stage2, by_task=bytask,
                n_below_half=below, n=len(recs)),
), open(os.path.join(OUT, "stats_retired_final.json"), "w"), indent=1, default=float)
print(f"\nwrote {OUT}/stats_retired_final.json, table_retired_summary.csv, "
      f"table_censoring.csv")
