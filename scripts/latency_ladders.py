#!/usr/bin/env python3
"""
latency_ladders.py -- the language-model frontier with a measured resource.

Everything in Section 8.4 so far prices language models by capacity or by
training tokens.  That was not a choice: the mixed-population fronts of
Section 8.4.1 do use measured latency, but they mix families, and the
within-family ladders of Section 8.4.3 hold the family fixed at the cost of
substituting parameter count for time.  Neither is the object the theory is
about, which is one comparable population priced by the resource a deployment
actually spends.

Both constraints can be met at once with data already in the corpus.  The
LLM-Perf leaderboard measures end-to-end latency for released model sizes on
fixed hardware, and the Open LLM Leaderboard scores those same checkpoints
benchmark by benchmark.  Restricting to one family, one device and one
precision leaves capacity as the only thing that varies, exactly as in the
capacity-ladder diagnostic -- but the horizontal axis is now a wall-clock
measurement rather than a proxy for one.

Qwen1.5 is the only family the leaderboard covers densely enough: seven
released sizes from 0.5B to 110B, spanning 3.4 e-folds of measured latency.
The identifiability rule of Section 5.2 is applied unchanged.

Reads   corpus_llm.csv
Writes  table_latladders.csv, stats_latladders.json
"""
import collections
import csv
import json
import math
import os
import re
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2, t as tdist

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260824)

DATA = os.environ.get("DATA", "data")
OUT = os.environ.get("OUT", "results")
B = int(os.environ.get("BOOT", 400))
MINFRONT = 6

# Benchmarks retired from the Open LLM Leaderboard as models reached them,
# against those introduced in v2 to restore headroom.  Fixed before fitting.
RETIRED = {"HellaSwag", "ARC-Challenge", "MMLU", "Winogrande", "GSM8K"}
HEADROOM = {"IFEval", "BBH", "MATH Lvl 5", "GPQA", "MuSR", "MMLU-PRO"}


def family(name):
    org, _, tail = name.partition("/")
    t = re.sub(r"[-_]?\d+\.?\d*x?\d*\.?\d*\s*[bBmM]\b", "", tail)
    t = re.sub(r"[-_]?\d+\.?\d*[bBmM]$", "", t)
    t = re.sub(r"[-_]?(base|hf|it|instruct|chat)$", "", t, flags=re.I)
    return f"{org}/{t}".lower().strip("-_ ")


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


def fit_fast(t, e, p0):
    lr, ee = np.log(np.asarray(t, float)), np.asarray(e, float)
    f = lambda lr, ei, lc, a: ei + np.exp(lc) * np.exp(-a * lr)
    hi = min(ee) * .999
    try:
        p, _ = curve_fit(f, lr, ee,
                         p0=[min(p0[0], hi * .9), math.log(max(p0[1], 1e-12)), p0[2]],
                         maxfev=20000, bounds=([0, -60, .01], [hi, 60, 8]))
    except Exception:
        return None
    return p[0], math.exp(p[1]), p[2]


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
    y, s2 = np.asarray(y, float), np.maximum(np.asarray(s2, float), 1e-12)
    k = len(y)
    if k == 1:
        se = math.sqrt(s2[0])
        return dict(k=1, mu=float(y[0]), se=se, tau=.0, I2=.0,
                    ci=[float(y[0] - 1.96 * se), float(y[0] + 1.96 * se)])
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


# ------------------------------------------------------------------ inputs --
cells = collections.defaultdict(dict)
for r in csv.DictReader(open(f"{DATA}/corpus_llm.csv")):
    if r["source"] != "llm-perf-task":
        continue
    try:
        t, Q, ch = float(r["latency_ms"]), float(r["Q"]), float(r["chance"])
    except (TypeError, ValueError):
        continue
    if t <= 0:
        continue
    e = 1 - (Q - ch) / (1 - ch)
    if not (0 < e < 1):
        continue
    task = r["task"].replace("LLM ", "")
    cells[(family(r["model"]), r["device"], r["precision"], task)][r["model"]] \
        = (t, e)

print("within-family latency ladders: one family, one device, one precision,\n"
      "capacity varying, priced by measured end-to-end latency\n")

table, ref = [], {}
for key, v in sorted(cells.items()):
    fam, dev, prec, task = key
    label = f"{fam.split('/')[-1]} | {dev} | {prec} | {task}"
    front = pareto(list(v.values()))
    row = dict(label=label, family=fam, device=dev, precision=prec, task=task,
               benchmark_class=("retired" if task in RETIRED else
                                "headroom" if task in HEADROOM else "other"),
               n_sizes=len(v), n_front=len(front))
    if len(front) >= 2:
        row["S_range"] = round(math.log(front[0][1] / front[-1][1]), 3)
        row["x_range"] = round(math.log(front[-1][0] / front[0][0]), 3)
        row["eps_min"] = round(front[-1][1], 5)
    if len(front) < MINFRONT:
        row["status"] = "excluded: front shorter than six points"
        table.append(row)
        continue
    t = np.array([p[0] for p in front], float)
    e = np.array([p[1] for p in front], float)
    f = fit_knee(t, e)
    if f is None or not (0 < f[2] < 8):
        row["status"] = "excluded: fit failed"
        table.append(row)
        continue
    ei, c, a, r2 = f
    pw = fit_power(t, e)
    rss_sat = float(((e - (ei + c * t ** (-a))) ** 2).sum())
    row.update(eps_inf=round(ei, 6), alpha=round(a, 4),
               lambda_star=round(a / 2, 4), R2=round(r2, 4))
    if pw is not None:
        row["d_aicc"] = round(float(aicc(pw[2], len(t), 2)
                                    - aicc(rss_sat, len(t), 3)), 2)
    if ei <= 1e-6 or ei >= .999 * e.min():
        row["status"] = "excluded: floor at boundary (degenerate)"
        table.append(row)
        continue
    row["status"] = "identified"
    lams, tries = [], 0
    while len(lams) < B and tries < 8 * B:
        tries += 1
        idx = RNG.integers(0, len(t), len(t))
        tt, ee = t[idx], e[idx]
        if len(np.unique(tt)) < 4:
            continue
        g = fit_fast(tt, ee, (ei, c, a))
        if g is None or not (0 < g[2] < 8):
            continue
        if g[0] <= 1e-6 or g[0] >= .999 * ee.min():
            continue
        lams.append(g[2] / 2)
    lams = np.array(lams) if len(lams) >= 30 else np.array([a / 2] * 30)
    row.update(boot_se=round(float(lams.std(ddof=1)), 4),
               boot_lo=round(float(np.percentile(lams, 2.5)), 4),
               boot_hi=round(float(np.percentile(lams, 97.5)), 4))
    table.append(row)
    ref[label] = dict(row, front=front)
    print(f"  {label:46s} |P|={len(t)} lam*={a / 2:.3f} "
          f"[{row['boot_lo']:.3f},{row['boot_hi']:.3f}] R2={r2:.3f} "
          f"dAICc={row.get('d_aicc', float('nan')):+.1f}  "
          f"{row['benchmark_class']}")

# ---------------------------------------------------------------- summary ---
cand = [r for r in table if r["n_front"] >= MINFRONT]
print(f"\n  {len(ref)} identified of {len(cand)} candidate ladders "
      f"({len({r['device'] for r in cand})} devices, "
      f"{len({r['task'] for r in cand})} benchmarks)")

by_class = {}
for cls in ("retired", "headroom"):
    sub = [r for r in cand if r["benchmark_class"] == cls]
    ident = [r for r in sub if r["status"] == "identified"]
    da = [r["d_aicc"] for r in sub if r.get("d_aicc") is not None]
    by_class[cls] = dict(n=len(sub), identified=len(ident),
                         saturating=int(sum(1 for x in da if x > 0)),
                         n_aicc=len(da),
                         d_aicc=float(np.median(da)) if da else None,
                         S_range=float(np.median([r.get("S_range", 0)
                                                  for r in sub])))
    print(f"  {cls:9s}: {len(ident)}/{len(sub)} identified, "
          f"saturating preferred {by_class[cls]['saturating']}/{len(da)}, "
          f"median dAICc {by_class[cls]['d_aicc']:+.2f}, "
          f"median span {by_class[cls]['S_range']:.2f} nats")

if ref:
    keys = sorted(ref)
    pool = dl([ref[k]["lambda_star"] for k in keys],
              [ref[k]["boot_se"] ** 2 for k in keys])
    ret = [k for k in keys if ref[k]["benchmark_class"] == "retired"]
    pool_r = dl([ref[k]["lambda_star"] for k in ret],
                [ref[k]["boot_se"] ** 2 for k in ret]) if ret else None
    bytask = collections.defaultdict(list)
    for k in keys:
        bytask[ref[k]["task"]].append(k)
    stage1 = {t: dl([ref[k]["lambda_star"] for k in ks],
                    [ref[k]["boot_se"] ** 2 for k in ks])
              for t, ks in sorted(bytask.items())}
    stage2 = dl([p["mu"] for p in stage1.values()],
                [p["se"] ** 2 for p in stage1.values()])
    print(f"\n  pooled lambda*, ladder as unit   {pool['mu']:.3f} "
          f"[{pool['ci'][0]:.3f},{pool['ci'][1]:.3f}]  tau={pool['tau']:.3f} "
          f"I2={pool['I2']:.0f}%")
    if pool_r:
        print(f"  retired benchmarks only          {pool_r['mu']:.3f} "
              f"[{pool_r['ci'][0]:.3f},{pool_r['ci'][1]:.3f}]  "
              f"(k={pool_r['k']}, I2={pool_r['I2']:.0f}%)")
    print(f"  two-stage, benchmark as unit     {stage2['mu']:.3f} "
          f"[{stage2['ci'][0]:.3f},{stage2['ci'][1]:.3f}]  (k={stage2['k']})")
    for t, p in stage1.items():
        print(f"    {t:16s} k={p['k']}  {p['mu']:.3f} "
              f"[{p['ci'][0]:.3f},{p['ci'][1]:.3f}]")
    below = sum(1 for k in keys if ref[k]["boot_hi"] < .5)
    print(f"  bootstrap interval entirely below 1/2: {below}/{len(keys)}")
    print(f"  median R2 {np.median([ref[k]['R2'] for k in keys]):.3f}; "
          f"AICc prefers saturating in "
          f"{sum(1 for k in keys if ref[k].get('d_aicc', 0) > 0)}/{len(keys)}")
else:
    pool = pool_r = stage2 = stage1 = None
    below = 0

cols = ["label", "family", "device", "precision", "task", "benchmark_class",
        "n_sizes", "n_front", "eps_min", "eps_inf", "alpha", "lambda_star",
        "boot_lo", "boot_hi", "boot_se", "R2", "S_range", "x_range", "d_aicc",
        "status"]
with open(f"{OUT}/table_latladders.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(table, key=lambda z: (z["status"] != "identified", z["label"])):
        w.writerow(r)

json.dump(dict(
    n_candidate=len(cand), n_identified=len(ref),
    n_devices=len({r["device"] for r in cand}),
    n_tasks=len({r["task"] for r in cand}),
    families=sorted({r["family"] for r in cand}),
    median_x_range=float(np.median([r["x_range"] for r in cand])),
    by_class=by_class, pooled=pool, pooled_retired=pool_r,
    by_task=stage1, two_stage=stage2, n_below_half=below,
    cells=[dict(label=k, task=ref[k]["task"], device=ref[k]["device"],
                benchmark_class=ref[k]["benchmark_class"],
                lam=ref[k]["lambda_star"], lo=ref[k]["boot_lo"],
                hi=ref[k]["boot_hi"], se=ref[k]["boot_se"], R2=ref[k]["R2"],
                eps_inf=ref[k]["eps_inf"], d_aicc=ref[k].get("d_aicc"),
                n_front=ref[k]["n_front"],
                points=[[float(a), float(b)] for a, b in ref[k]["front"]])
           for k in sorted(ref)],
), open(f"{OUT}/stats_latladders.json", "w"), indent=1, default=float)
print(f"\nwrote {OUT}/table_latladders.csv, {OUT}/stats_latladders.json")
