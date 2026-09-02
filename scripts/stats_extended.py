#!/usr/bin/env python3
"""
stats_extended.py  --  fit and pool the canonical price across every domain.

Combines the primary vision-inference corpus (corpus.csv, mai_data.csv) with
the extended cross-domain corpus (corpus_extended.csv: semantic segmentation,
object detection, instance segmentation, speech recognition, LLM inference)
and applies the same identifiability rule, bootstrap and random-effects
machinery to all of them.

Outputs
    table_extended.csv    every candidate cell, identified or not, with the
                          reason for exclusion
    stats_extended.json   pooled prices overall and by task, heterogeneity,
                          the test of the nonparametric bound, and the LLM
                          diagnostics
"""
import collections
import csv
import json
import math
import os
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t as tdist, norm, chi2

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260824)

_D = os.environ.get("DATA", "data")
_OUT = os.environ.get("OUT", "results")
D, OUT = _D, _OUT
os.makedirs(OUT, exist_ok=True)

B = int(os.environ.get("BOOT", 500))
MINFRONT = 6


# ------------------------------------------------------------------ shared --
def pareto(pts):
    pts = sorted(pts)
    out, best = [], np.inf
    for t, e in pts:
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
                             bounds=([0, -40, .01], [min(ee) * .999, 40, 8]))
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
                         maxfev=20000, bounds=([0, -40, .01], [hi, 40, 8]))
    except Exception:
        return None
    return (p[0], math.exp(p[1]), p[2], np.nan)


def fit_power(t, e):
    lr, ee = np.log(np.asarray(t, float)), np.asarray(e, float)
    f = lambda lr, lc, a: np.exp(lc) * np.exp(-a * lr)
    try:
        p, _ = curve_fit(f, lr, ee, p0=[0., .5], maxfev=400000,
                         bounds=([-40, .0], [40, 8]))
    except Exception:
        return None
    return math.exp(p[0]), p[1], float(((ee - f(lr, *p)) ** 2).sum())


def rss_knee(t, e, par):
    lr, ee = np.log(np.asarray(t, float)), np.asarray(e, float)
    return float(((ee - (par[0] + par[1] * np.exp(-par[2] * lr))) ** 2).sum())


def aicc(rss, n, k):
    return np.nan if n - k - 1 <= 0 else \
        n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


# ------------------------------------------------------------------- cells --
cells = collections.defaultdict(list)
meta = {}


def add(key, task, group, t, e):
    cells[key].append((t, e))
    meta[key] = (task, group)


for x in csv.DictReader(open(f"{D}/corpus.csv")):
    Q, ch, t = float(x["Q"]), float(x["chance"]), float(x["latency_ms"])
    if t <= 0:
        continue
    e = 1 - (Q - ch) / (1 - ch)
    if 0 < e < 1:
        add((x["source"], x["cell"], x["task"]), x["task"],
            "Primary vision inference", t, e)

if os.path.exists(f"{D}/mai_data.csv"):
    for x in csv.DictReader(open(f"{D}/mai_data.csv")):
        t, p = float(x["runtime_ms"]), float(x["psnr"])
        if t > 0:
            add(("mai", x["cell"], "Super-resolution"), "Super-resolution",
                "Primary vision inference", t, 10 ** (-p / 10))

for x in csv.DictReader(open(f"{D}/corpus_extended.csv")):
    Q, ch, t = float(x["Q"]), float(x["chance"]), float(x["latency_ms"])
    if t <= 0:
        continue
    e = 1 - (Q - ch) / (1 - ch)
    if 0 < e < 1:
        group = ("LLM inference" if x["task"] == "LLM inference"
                 else "Extended cross-domain")
        add((x["source"], x["cell"], x["task"]), x["task"], group, t, e)

print(f"candidate cells: {len(cells)}")


# ------------------------------------------------------------------- fits --
ref, table = {}, []
for k, v in sorted(cells.items()):
    task, group = meta[k]
    P = pareto(v)
    row = dict(source=k[0], cell=k[1], task=task, group=group,
               n=len(v), n_front=len(P))
    if len(P) < MINFRONT:
        row.update(status="excluded: front shorter than six points")
        table.append(row)
        continue
    f = fit_knee([p[0] for p in P], [p[1] for p in P])
    if f is None:
        row.update(status="excluded: fit failed")
        table.append(row)
        continue
    ei, c, a, r2 = f
    row.update(eps_inf=round(ei, 5), alpha=round(a, 4),
               lambda_star=round(a / 2, 4), R2=round(r2, 4),
               S_range=round(math.log(max(p[1] for p in P) /
                                      min(p[1] for p in P)), 3),
               x_range=round(math.log(max(p[0] for p in P) /
                                      min(p[0] for p in P)), 3))
    if ei <= 1e-6 or ei >= 0.999 * min(p[1] for p in P):
        row.update(status="excluded: floor at boundary (degenerate)")
        table.append(row)
        continue
    row.update(status="identified")
    table.append(row)
    ref[k] = dict(P=P, eps_inf=ei, c=c, alpha=a, r2=r2, lam=a / 2,
                  task=task, group=group)

print(f"identified cells: {len(ref)}")


def short(k):
    src, cell, task = k
    if src == "timm":
        p = [s.strip() for s in cell.split("|")]
        return f"{p[1]} / {p[2].replace('PyTorch 2.4.0 ', '').replace('PyTorch 2.9.1 ', '')} {p[3]}"
    if src in ("mmseg", "mmdet"):
        p = [s.strip() for s in cell.split("|")]
        return f"{p[1]} / {p[2]} / {task.split()[0].lower()}"
    if src.startswith("open-asr"):
        return "ASR / " + cell.split("|")[1].strip()
    if src.startswith("llm-perf"):
        return "LLM / " + cell.split("|")[2].strip()
    return f"CPU ONNX / {task}"


# -------------------------------------------------------------- bootstrap --
boot = {}
for k, d in ref.items():
    P = np.array(d["P"])
    lams = []
    tries = 0
    while len(lams) < B and tries < 8 * B:
        tries += 1
        idx = RNG.integers(0, len(P), len(P))
        tt, ee = P[idx, 0], P[idx, 1]
        if len(np.unique(tt)) < 4:
            continue
        f = fit_fast(tt, ee, (d["eps_inf"], d["c"], d["alpha"]))
        if f is None or not (0 < f[2] < 8):
            continue
        if f[0] <= 1e-6 or f[0] >= 0.999 * ee.min():
            continue
        lams.append(f[2] / 2)
    lams = np.array(lams)
    if len(lams) < 50:
        lams = np.array([d["lam"]] * 50)
    print(f"  boot {short(k):46s} lam*={d['lam']:.3f} "
          f"[{np.percentile(lams, 2.5):.3f},{np.percentile(lams, 97.5):.3f}] "
          f"({len(lams)} draws)", flush=True)
    boot[k] = dict(se=float(lams.std(ddof=1)),
                   lo=float(np.percentile(lams, 2.5)),
                   hi=float(np.percentile(lams, 97.5)),
                   n=len(lams))


# ------------------------------------------------------- random effects ----
def pool(keys):
    y = np.array([ref[k]["lam"] for k in keys])
    s2 = np.array([boot[k]["se"] ** 2 for k in keys])
    kk = len(y)
    if kk == 1:
        return dict(k=1, mu=float(y[0]), se=float(math.sqrt(s2[0])),
                    ci=[float(y[0] - 1.96 * math.sqrt(s2[0])),
                        float(y[0] + 1.96 * math.sqrt(s2[0]))],
                    tau=0.0, Q=0.0, Q_df=0, Q_p=1.0, I2=0.0,
                    pi=[float(y[0]), float(y[0])])
    w = 1 / np.maximum(s2, 1e-12)
    mu_fe = float((w * y).sum() / w.sum())
    Q = float((w * (y - mu_fe) ** 2).sum())
    df = kk - 1
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / C)
    wr = 1 / (np.maximum(s2, 1e-12) + tau2)
    mu = float((wr * y).sum() / wr.sum())
    se = float(math.sqrt(1 / wr.sum()))
    tc = float(tdist.ppf(0.975, max(kk - 2, 1)))
    return dict(k=kk, mu=mu, se=se, ci=[mu - 1.96 * se, mu + 1.96 * se],
                tau=math.sqrt(tau2), Q=Q, Q_df=df,
                Q_p=float(1 - chi2.cdf(Q, df)),
                I2=max(0.0, 100 * (Q - df) / Q) if Q > 0 else 0.0,
                pi=[mu - tc * math.sqrt(tau2 + se ** 2),
                    mu + tc * math.sqrt(tau2 + se ** 2)])


groups = collections.defaultdict(list)
tasks = collections.defaultdict(list)
for k in ref:
    groups[ref[k]["group"]].append(k)
    tasks[ref[k]["task"]].append(k)

allk = sorted(ref, key=lambda k: ref[k]["lam"])
P_all = pool(allk)
P_by_group = {g: pool(ks) for g, ks in groups.items()}
P_by_task = {t: pool(ks) for t, ks in tasks.items() if len(ks) >= 2}

print(f"\nALL {P_all['k']} cells: pooled {P_all['mu']:.3f} "
      f"CI [{P_all['ci'][0]:.3f},{P_all['ci'][1]:.3f}] tau={P_all['tau']:.3f} "
      f"PI [{P_all['pi'][0]:.3f},{P_all['pi'][1]:.3f}] "
      f"I2={P_all['I2']:.0f}% Q_p={P_all['Q_p']:.1e}")
for g, p in sorted(P_by_group.items()):
    print(f"  {g:26s} k={p['k']:2d} pooled {p['mu']:.3f} "
          f"CI [{p['ci'][0]:.3f},{p['ci'][1]:.3f}] tau={p['tau']:.3f} "
          f"I2={p['I2']:.0f}%")
print()
for t, p in sorted(P_by_task.items(), key=lambda z: z[1]["mu"]):
    print(f"  {t:24s} k={p['k']:2d} pooled {p['mu']:.3f} "
          f"CI [{p['ci'][0]:.3f},{p['ci'][1]:.3f}]")


# ------------------------------------------------------------ the bound ----
above = [k for k in allk if boot[k]["lo"] > 0.5]
below = [k for k in allk if boot[k]["hi"] < 0.5]
print(f"\nbound lambda* < 1/2: {len(below)}/{len(allk)} cells entirely below, "
      f"{len(above)} entirely above")
for k in above:
    print(f"   ABOVE  {short(k):46s} lam*={ref[k]['lam']:.3f} "
          f"[{boot[k]['lo']:.3f},{boot[k]['hi']:.3f}]")


# ----------------------------------------------------------------- AICc ----
aic = {}
for k in allk:
    P = np.array(ref[k]["P"])
    t, e, n = P[:, 0], P[:, 1], len(P)
    rs = rss_knee(t, e, (ref[k]["eps_inf"], ref[k]["c"], ref[k]["alpha"]))
    pw = fit_power(t, e)
    if pw is None:
        continue
    d = aicc(pw[2], n, 2) - aicc(rs, n, 3)
    aic[k] = float(d)
da = np.array(list(aic.values()))
print(f"\nAICc: saturating preferred in {(da > 0).sum()}/{len(da)} cells, "
      f"median {np.median(da):.1f}, min {da.min():.1f}")


# ------------------------------------------------------------------ LLM ----
llm = {}
for r in table:
    if r["task"] != "LLM inference":
        continue
    llm[r["cell"]] = {kk: r.get(kk) for kk in
                      ("n", "n_front", "eps_inf", "alpha", "lambda_star",
                       "R2", "S_range", "x_range", "status")}
print("\nLLM cells:")
for c, r in llm.items():
    print(f"  {c:34s} n={r['n']:5d} |P|={r['n_front']:3d} "
          f"alpha={r.get('alpha')} lam*={r.get('lambda_star')} "
          f"R2={r.get('R2')} S-range={r.get('S_range')} -- {r['status']}")

vis = [r for r in table if r["status"] == "identified"
       and r["group"] != "LLM inference"]
print(f"\nmedian S-range: vision/speech {np.median([r['S_range'] for r in vis]):.2f} "
      f"nats;  LLM {np.median([r['S_range'] for r in table if r['task'] == 'LLM inference' and 'S_range' in r]):.2f} nats")


# --------------------------------------------------------------- outputs ---
with open(f"{OUT}/table_extended.csv", "w", newline="") as fh:
    cols = ["source", "cell", "task", "group", "n", "n_front", "eps_inf",
            "alpha", "lambda_star", "boot_lo", "boot_hi", "boot_se", "R2",
            "S_range", "x_range", "d_aicc", "status"]
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in table:
        k = (r["source"], r["cell"], r["task"])
        if k in boot:
            r = dict(r, boot_lo=round(boot[k]["lo"], 4),
                     boot_hi=round(boot[k]["hi"], 4),
                     boot_se=round(boot[k]["se"], 4),
                     d_aicc=round(aic.get(k, float("nan")), 2))
        w.writerow(r)

json.dump(dict(
    n_candidate=len(cells), n_identified=len(ref),
    all=P_all, by_group=P_by_group, by_task=P_by_task,
    cells=[dict(label=short(k), task=ref[k]["task"], group=ref[k]["group"],
                lam=ref[k]["lam"], lo=boot[k]["lo"], hi=boot[k]["hi"],
                se=boot[k]["se"], eps_inf=ref[k]["eps_inf"],
                alpha=ref[k]["alpha"], R2=ref[k]["r2"],
                n_front=len(ref[k]["P"]), d_aicc=aic.get(k))
           for k in allk],
    n_below_half=len(below), n_above_half=len(above),
    above_half=[short(k) for k in above],
    aicc_wins=int((da > 0).sum()), aicc_n=int(len(da)),
    aicc_median=float(np.median(da)), aicc_min=float(da.min()),
    llm=llm,
    S_range_vis=float(np.median([r["S_range"] for r in vis])),
    S_range_llm=float(np.median([r["S_range"] for r in table
                                 if r["task"] == "LLM inference" and "S_range" in r])),
), open(f"{OUT}/stats_extended.json", "w"), indent=1)
print("\nwrote table_extended.csv, stats_extended.json")
