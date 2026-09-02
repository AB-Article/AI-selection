#!/usr/bin/env python3
"""
latency_final.py -- the three tests that decide whether the latency ladders
support the theory or merely tolerate it.

Section 8.4.5 leaves three questions open, and each has an answer in the data
already collected.

  Q1  Model selection.  AICc favours the saturating law in only 46 of 125
      ladders, but a six-point front has almost no power to prefer a third
      parameter, so a per-ladder verdict is the wrong unit.  Fitted to
      independent data, AICc differences add, so the joint evidence over the
      34 family-by-benchmark units is their sum -- with the ladders inside a
      unit averaged first, since they are one measurement repeated.
  Q2  Stability.  lambda* = alpha/2 is a property of a frontier.  A benchmark
      measured on two or three model families, on different GPUs and different
      inference backends, should therefore return the same price if the theory
      describes something real about the benchmark, and should not if the
      number is an artefact of the fitting.
  Q3  The ceiling.  Three of these benchmarks state a human or expert accuracy
      in their source papers.  None of the three numbers enters any fit, and no
      fitted floor should fall below one: a model family cannot asymptote past
      the noise of the labels.

Reads   results/table_latladders.csv, results/stats_latladders.json
Writes  stats_latfinal.json
"""
import collections
import csv
import json
import math
import os

import numpy as np
from scipy.stats import chi2, wilcoxon, binomtest, spearmanr

DATA = os.environ.get("DATA", "data")
OUT = os.environ.get("OUT", "results")
RNG = np.random.default_rng(20260824)
NBOOT = int(os.environ.get("CLUSTER_BOOT", 4000))

# published human or expert accuracies; none enters any fit
CEILING = {"HellaSwag": (0.956, 0.25),    # Zellers et al. 2019
           "Winogrande": (0.940, 0.50),   # Sakaguchi et al. 2020
           "MMLU": (0.898, 0.25)}         # Hendrycks et al. 2021, expert level


def eps_ceiling(task):
    if task not in CEILING:
        return None
    h, ch = CEILING[task]
    return max(1e-6, 1 - (h - ch) / (1 - ch))


def dl(y, s2):
    y, s2 = np.asarray(y, float), np.maximum(np.asarray(s2, float), 1e-12)
    k = len(y)
    if k == 1:
        return dict(k=1, mu=float(y[0]), se=math.sqrt(s2[0]), tau=.0, I2=.0,
                    Q_p=1.0)
    w = 1 / s2
    mufe = (w * y).sum() / w.sum()
    Q = float((w * (y - mufe) ** 2).sum())
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0., (Q - (k - 1)) / C)
    wr = 1 / (s2 + tau2)
    mu = float((wr * y).sum() / wr.sum())
    return dict(k=k, mu=mu, se=float(math.sqrt(1 / wr.sum())),
                tau=math.sqrt(tau2), Q=Q,
                I2=max(0., 100 * (Q - (k - 1)) / Q) if Q > 0 else .0,
                Q_p=float(1 - chi2.cdf(Q, k - 1)))


rows = [r for r in csv.DictReader(open(f"{OUT}/table_latladders.csv"))]
for r in rows:
    for k in ("lambda_star", "boot_se", "d_aicc", "R2", "eps_min", "eps_inf",
              "S_range"):
        r[k] = float(r[k]) if r[k] not in ("", None) else None
    r["n_front"] = int(r["n_front"])
ident = [r for r in rows if r["status"] == "identified"]
cand = [r for r in rows if r["n_front"] >= 6]

units = collections.defaultdict(list)
for r in ident:
    units[(r["family"], r["task"])].append(r)

# ============================================================ Q1 selection ==
print("Q1  joint model selection over the family-by-benchmark units\n")
unit_d = {k: float(np.mean([r["d_aicc"] for r in v if r["d_aicc"] is not None]))
          for k, v in sorted(units.items())
          if any(r["d_aicc"] is not None for r in v)}
vals = np.array(list(unit_d.values()))
total = float(vals.sum())
pos = int((vals > 0).sum())
sign_p = float(binomtest(pos, len(vals), .5).pvalue)
wil = wilcoxon(vals, alternative="greater") if len(vals) > 5 else None
boot = np.array([np.mean(RNG.choice(vals, len(vals), replace=True))
                 for _ in range(NBOOT)])
print(f"  {len(vals)} units; summed dAICc = {total:+.1f}, "
      f"mean {vals.mean():+.2f} [{np.percentile(boot, 2.5):+.2f},"
      f"{np.percentile(boot, 97.5):+.2f}]")
print(f"  saturating favoured in {pos}/{len(vals)} units "
      f"(sign test p = {sign_p:.3f}"
      + (f", Wilcoxon p = {wil.pvalue:.3f})" if wil else ")"))
by_cls = {}
for cls in ("retired", "headroom"):
    v = np.array([d for k, d in unit_d.items()
                  if units[k][0]["benchmark_class"] == cls])
    if len(v):
        by_cls[cls] = dict(n=len(v), total=float(v.sum()),
                           mean=float(v.mean()),
                           pos=int((v > 0).sum()))
        print(f"  {cls:9s}: {len(v)} units, summed {v.sum():+.1f}, "
              f"favourable in {(v > 0).sum()}/{len(v)}")
# is the evidence ordered by how far the front descends, as elsewhere?
sm = [-math.log(np.median([r["eps_min"] for r in units[k]])) for k in unit_d]
rho = spearmanr(sm, list(unit_d.values()))
print(f"  rho(resolved information, unit dAICc) = {rho.statistic:+.2f} "
      f"(p = {rho.pvalue:.3f})")

# ========================================================== Q2 stability ====
print("\nQ2  is the price stable across families, GPUs and backends?\n")
byfam = collections.defaultdict(dict)
for (fam, task), v in units.items():
    p = dl([r["lambda_star"] for r in v], [r["boot_se"] ** 2 for r in v])
    byfam[task][fam] = p
shared = {t: f for t, f in byfam.items() if len(f) >= 2}
across = []
for t, f in sorted(shared.items()):
    p = dl([x["mu"] for x in f.values()], [x["se"] ** 2 for x in f.values()])
    across.append(dict(task=t, k=p["k"], I2=p["I2"], Q_p=p["Q_p"], mu=p["mu"],
                       spread=float(max(x["mu"] for x in f.values())
                                    - min(x["mu"] for x in f.values()))))
    print(f"  {t:14s} {p['k']} families  lambda* {p['mu']:.3f}  "
          f"spread {across[-1]['spread']:.3f}  I2 {p['I2']:3.0f}%  "
          f"Q p = {p['Q_p']:.3f}")
agree = sum(1 for a in across if a["Q_p"] > .05)
print(f"  homogeneous across families at the 5% level in "
      f"{agree}/{len(across)} benchmarks")

within = []
for (fam, task), v in sorted(units.items()):
    if len(v) < 3:
        continue
    p = dl([r["lambda_star"] for r in v], [r["boot_se"] ** 2 for r in v])
    within.append(dict(family=fam, task=task, k=p["k"], I2=p["I2"],
                       Q_p=p["Q_p"]))
wagree = sum(1 for w in within if w["Q_p"] > .05)
print(f"  within a family and benchmark, across hardware conditions and "
      f"context lengths: homogeneous in {wagree}/{len(within)} units, "
      f"median I2 {np.median([w['I2'] for w in within]):.0f}%")

# ========================================================== Q3 the ceiling ==
print("\nQ3  do the fitted floors respect the published ceilings?\n")
anchor = []
for r in ident:
    ec = eps_ceiling(r["task"])
    if ec is None or r["eps_inf"] is None or r["eps_min"] is None:
        continue
    anchor.append(dict(label=r["label"], task=r["task"], family=r["family"],
                       eps_inf=r["eps_inf"], eps_min=r["eps_min"], eps_c=ec,
                       A=ec / r["eps_min"], respects=r["eps_inf"] >= ec))
if anchor:
    viol = [a for a in anchor if not a["respects"]]
    pmf = np.array([1.0])
    for a in anchor:
        pmf = np.convolve(pmf, [1 - min(a["A"], 1), min(a["A"], 1)])
    p_left = float(pmf[:len(viol) + 1].sum())
    exp = float(sum(min(a["A"], 1) for a in anchor))
    print(f"  {len(anchor)} identified ladders on "
          f"{len({a['task'] for a in anchor})} benchmarks with a published "
          f"human or expert accuracy")
    print(f"  floors respecting eps_inf >= eps_ceiling: "
          f"{len(anchor) - len(viol)}/{len(anchor)}")
    print(f"  expected violations if the floor were uninformative {exp:.1f}; "
          f"observed {len(viol)}; P(V <= obs) = {p_left:.2e}")
    for t in sorted({a["task"] for a in anchor}):
        s = [a for a in anchor if a["task"] == t]
        print(f"    {t:12s} n={len(s):2d} ceiling {s[0]['eps_c']:.3f}  "
              f"floors {min(a['eps_inf'] for a in s):.3f}"
              f"-{max(a['eps_inf'] for a in s):.3f}  "
              f"violations {sum(1 for a in s if not a['respects'])}")
else:
    p_left = exp = None

json.dump(dict(
    selection=dict(n_units=len(vals), total=total, mean=float(vals.mean()),
                   ci=[float(np.percentile(boot, 2.5)),
                       float(np.percentile(boot, 97.5))],
                   favourable=pos, sign_p=sign_p,
                   wilcoxon_p=float(wil.pvalue) if wil else None,
                   by_class=by_cls,
                   rho=[float(rho.statistic), float(rho.pvalue)],
                   units={f"{a} | {b}": d for (a, b), d in unit_d.items()}),
    stability=dict(across_families=across, n_homogeneous=agree,
                   n_shared=len(across), within=within,
                   n_within_homogeneous=wagree,
                   median_within_I2=float(np.median([w["I2"] for w in within]))
                   if within else None),
    anchor=dict(n=len(anchor), violations=len(viol) if anchor else None,
                expected=exp, p_left=p_left, ceilings=CEILING,
                eps_ceiling={t: eps_ceiling(t) for t in CEILING},
                rows=anchor),
), open(f"{OUT}/stats_latfinal.json", "w"), indent=1, default=float)
print(f"\nwrote {OUT}/stats_latfinal.json")
