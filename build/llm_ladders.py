#!/usr/bin/env python3
"""
llm_ladders.py  --  does the language-model failure come from the theory, or
from an unpriced hidden variable in the population?

Section 12.3 leaves three readings open.  This script separates two of them.

The heterogeneous LLM cells put models from different families, trained on
different corpora for different numbers of tokens, on one Pareto front, and
price them by decode latency -- which at batch one is bandwidth-bound and is
therefore close to a proxy for parameter count.  Training tokens, which is what
actually buys quality, are never priced.  A front drawn over such a population
is an *envelope* over a family of frontiers indexed by that hidden variable,
and an envelope is steeper than its members.  That predicts exactly what we
measured: alpha-hat inflated, prices above one half, poor model selection.

The test holds the hidden variable fixed.  Inside one model family the
tokeniser, the corpus, the recipe and the vintage are common, and capacity is
the only thing that varies, so parameter count is the whole resource.  If the
envelope reading is right, the within-family fronts should behave like the
vision cells: clean saturating fits, AICc favouring the saturating law, and
prices well below the values the mixed fronts produce.  If they misbehave in
the same way, the problem is not the population.

Test 2 checks the envelope mechanism directly, by asking whether release date
predicts where a model sits relative to the fitted frontier of its cell.  Under
the envelope reading later models sit below the front their size would imply,
because they saw more tokens; under a genuine frontier, date should carry no
information once latency is accounted for.

Reads   data/huggingface_v1.csv, data/huggingface_v2.csv, data/corpus_llm.csv
Writes  table_ladders.csv, stats_ladders.json
"""
import collections
import csv
import datetime as dt
import json
import math
import os
import re
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2, t as tdist, spearmanr

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260824)

DATA = os.environ.get("DATA", "data")
OUT = os.environ.get("OUT", "results")
os.makedirs(OUT, exist_ok=True)
B = int(os.environ.get("BOOT", 400))
MINFRONT = 6

V1_TASKS = {"arc": ("ARC-Challenge", .25), "hellaswag": ("HellaSwag", .25),
            "mmlu": ("MMLU", .25), "winogrande": ("Winogrande", .50),
            "gsm8k": ("GSM8K", .0)}
V2_TASKS = {"ifeval": "IFEval", "bbh": "BBH", "math": "MATH Lvl 5",
            "gpqa": "GPQA", "musr": "MuSR", "mmlu_pro": "MMLU-PRO"}


# --------------------------------------------------------------- machinery --
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


def dl(y, s2):
    y = np.asarray(y, float)
    s2 = np.maximum(np.asarray(s2, float), 1e-12)
    k = len(y)
    if k == 0:
        return None
    if k == 1:
        se = math.sqrt(s2[0])
        return dict(k=1, mu=float(y[0]), se=se, tau=.0, I2=.0, Q_p=1.,
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


# ------------------------------------------------------------------ inputs --
def family(name):
    org, _, tail = name.partition("/")
    n = re.sub(r"[-_]?\d+\.?\d*x?\d*\.?\d*\s*[bBmM]\b", "", tail)
    n = re.sub(r"[-_]?\d+\.?\d*[bBmM]$", "", n)
    n = re.sub(r"[-_]?(base|hf|it|instruct|chat)$", "", n, flags=re.I)
    return f"{org}/{n}".lower().strip("-_ ")


params, date, kind, scores = {}, {}, {}, collections.defaultdict(dict)

for r in csv.DictReader(open(f"{DATA}/huggingface_v2.csv")):
    m = r["model_name"].strip().lower()
    try:
        p = float(r["metadata_params_billions"])
        if p > 0:
            params.setdefault(m, p)
    except (TypeError, ValueError):
        pass
    kind.setdefault(m, r["model_type"])
    date.setdefault(m, r.get("metadata_submission_date", "") or
                    r.get("metadata_upload_date", ""))
    for stem, label in V2_TASKS.items():
        try:
            v = float(r[f"evaluations_{stem}_normalized_score"])
        except (TypeError, ValueError, KeyError):
            continue
        if v > 0:
            scores[m].setdefault((label, .0), v / 100.)

for r in csv.DictReader(open(f"{DATA}/huggingface_v1.csv")):
    m = r["model"].strip().lower()
    kind.setdefault(m, r["type"])
    date.setdefault(m, r.get("date", ""))
    try:
        p = float(r["params_b"])
        if p > 0:
            params.setdefault(m, p)          # integer-rounded; v2 wins
    except (TypeError, ValueError):
        pass
    for col, (label, ch) in V1_TASKS.items():
        try:
            v = float(r[col])
        except (TypeError, ValueError, KeyError):
            continue
        if v > 0:
            scores[m].setdefault((label, ch), v / 100.)

BASE = ("pretrained", "base", "continuously")
ladder = collections.defaultdict(list)
for m, sc in scores.items():
    if m not in params:
        continue
    k = kind.get(m, "").lower()
    if not any(b in k for b in BASE) and k.strip() != "":
        continue
    for (label, ch), Q in sc.items():
        e = 1 - (Q - ch) / (1 - ch)
        if 0 < e < 1:
            ladder[(family(m), label)].append((params[m], e, m))

# ============================================================ test 1 ========
print("TEST 1  within-family ladders: quality against capacity, "
      "recipe held fixed\n")
table, ref = [], {}
for (fam, task), v in sorted(ladder.items()):
    # one point per capacity: the best model of that size in the family
    by_p = {}
    for p, e, m in v:
        if p not in by_p or e < by_p[p][0]:
            by_p[p] = (e, m)
    pts = [(p, e) for p, (e, _) in by_p.items()]
    if len(pts) < MINFRONT:
        continue
    P = pareto(pts)
    row = dict(family=fam, task=task, n_sizes=len(pts), n_front=len(P))
    if len(P) >= 2:
        row["S_range"] = round(math.log(max(p[1] for p in P) /
                                        min(p[1] for p in P)), 3)
        row["x_range"] = round(math.log(max(p[0] for p in P) /
                                        min(p[0] for p in P)), 3)
    if len(P) < MINFRONT:
        row["status"] = "excluded: front shorter than six points"
        table.append(row)
        continue
    f = fit_knee([p[0] for p in P], [p[1] for p in P])
    if f is None or not (0 < f[2] < 8):
        row["status"] = "excluded: fit failed"
        table.append(row)
        continue
    ei, c, a, r2 = f
    row.update(eps_inf=round(ei, 6), alpha=round(a, 4),
               lambda_star=round(a / 2, 4), R2=round(r2, 4))
    if ei <= 1e-6 or ei >= .999 * min(p[1] for p in P):
        row["status"] = "excluded: floor at boundary (degenerate)"
        table.append(row)
        continue
    row["status"] = "identified"
    table.append(row)
    ref[(fam, task)] = dict(P=P, eps_inf=ei, c=c, alpha=a, lam=a / 2, r2=r2)

boot, aic = {}, {}
for k, d in ref.items():
    P = np.array(d["P"])
    lams, tries = [], 0
    while len(lams) < B and tries < 12 * B:
        tries += 1
        idx = RNG.integers(0, len(P), len(P))
        tt, ee = P[idx, 0], P[idx, 1]
        if len(np.unique(tt)) < 4:
            continue
        f = fit_fast(tt, ee, (d["eps_inf"], d["c"], d["alpha"]))
        if f is None or not (0 < f[2] < 8):
            continue
        if f[0] <= 1e-6 or f[0] >= .999 * ee.min():
            continue
        lams.append(f[2] / 2)
    lams = np.array(lams) if len(lams) >= 30 else np.array([d["lam"]] * 30)
    boot[k] = dict(se=float(lams.std(ddof=1)),
                   lo=float(np.percentile(lams, 2.5)),
                   hi=float(np.percentile(lams, 97.5)))
    pw = fit_power(P[:, 0], P[:, 1])
    if pw is not None:
        aic[k] = float(aicc(pw[2], len(P), 2) -
                       aicc(rss_knee(P[:, 0], P[:, 1],
                                     (d["eps_inf"], d["c"], d["alpha"])), len(P), 3))
    print(f"  {fam_task:52s}".replace("fam_task", "") if False else
          f"  {k[0] + ' / ' + k[1]:52s} lam*={d['lam']:.3f} "
          f"[{boot[k]['lo']:.3f},{boot[k]['hi']:.3f}] |P|={len(P)} "
          f"R2={d['r2']:.3f} dAICc={aic.get(k, float('nan')):+.1f}", flush=True)

for r in table:
    k = (r["family"], r["task"])
    if k in boot:
        r.update(boot_lo=round(boot[k]["lo"], 4), boot_hi=round(boot[k]["hi"], 4),
                 boot_se=round(boot[k]["se"], 4),
                 d_aicc=round(aic.get(k, float("nan")), 2))

cand = len([r for r in table if r["n_front"] >= MINFRONT])
pool = dl([ref[k]["lam"] for k in sorted(ref)],
          [boot[k]["se"] ** 2 for k in sorted(ref)]) if ref else None
byfam = collections.defaultdict(list)
for k in ref:
    byfam[k[0]].append(k)
stage1 = {f: dl([ref[k]["lam"] for k in ks], [boot[k]["se"] ** 2 for k in ks])
          for f, ks in sorted(byfam.items())}
stage2 = dl([p["mu"] for p in stage1.values()],
            [p["se"] ** 2 for p in stage1.values()]) if stage1 else None

if pool:
    lam = np.array([ref[k]["lam"] for k in ref])
    r2s = np.array([ref[k]["r2"] for k in ref])
    da = np.array([v for v in aic.values() if not math.isnan(v)])
    print(f"\n  {len(ref)} identified of {cand} candidate ladders "
          f"in {len(byfam)} families")
    print(f"  pooled lambda* {pool['mu']:.3f} "
          f"[{pool['ci'][0]:.3f},{pool['ci'][1]:.3f}] tau={pool['tau']:.3f} "
          f"I2={pool['I2']:.0f}%")
    if stage2:
        print(f"  two-stage (family as unit, k={stage2['k']}): {stage2['mu']:.3f} "
              f"[{stage2['ci'][0]:.3f},{stage2['ci'][1]:.3f}]")
    print(f"  median lambda* {np.median(lam):.3f}   median R2 {np.median(r2s):.3f}")
    print(f"  AICc prefers saturating in {(da > 0).sum()}/{len(da)}, "
          f"median {np.median(da):+.1f}")
    print(f"  cells with bootstrap interval entirely below 1/2: "
          f"{sum(1 for k in ref if boot[k]['hi'] < .5)}/{len(ref)}")

# ============================================================ test 2 ========
print("\nTEST 2  does release date predict position relative to the front?\n")
cells = collections.defaultdict(list)
for x in csv.DictReader(open(f"{DATA}/corpus_llm.csv")):
    if x["source"] != "llm-perf-task":
        continue
    Q, ch, t = float(x["Q"]), float(x["chance"]), float(x["latency_ms"])
    if t <= 0:
        continue
    e = 1 - (Q - ch) / (1 - ch)
    if 0 < e < 1:
        cells[x["cell"]].append((t, e, x["model"].strip().lower()))


def as_days(s):
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return (dt.datetime.strptime(s[:19] if "T" in s else s[:10], fmt)
                    - dt.datetime(2022, 1, 1)).days
        except (ValueError, TypeError):
            continue
    return None


vintage = []
for k, v in sorted(cells.items()):
    pts = [(t, e) for t, e, _ in v]
    P = pareto(pts)
    if len(P) < MINFRONT:
        continue
    f = fit_knee([p[0] for p in P], [p[1] for p in P])
    if f is None:
        continue
    ei, c, a, _ = f
    resid, days = [], []
    seen = set()
    for t, e, m in v:
        if m in seen:
            continue
        d = as_days(date.get(m, ""))
        if d is None:
            continue
        seen.add(m)
        pred = ei + c * t ** (-a)
        resid.append(math.log(max(e, 1e-12) / max(pred, 1e-12)))
        days.append(d)
    if len(resid) < 8:
        continue
    rho = spearmanr(days, resid)
    vintage.append(dict(cell=k, n=len(resid), rho=float(rho.statistic),
                        p=float(rho.pvalue)))
    print(f"  {k:44s} n={len(resid):3d} rho={rho.statistic:+.2f} "
          f"p={rho.pvalue:.1e}")

if vintage:
    rr = np.array([v["rho"] for v in vintage])
    sig = sum(1 for v in vintage if v["p"] < .05 and v["rho"] < 0)
    print(f"\n  median rho(date, log residual) = {np.median(rr):+.2f} over "
          f"{len(rr)} cells; {sig} significantly negative at 5%")
    print("  negative rho means later models sit BELOW the frontier their "
          "latency implies")

# ----------------------------------------------------------------- outputs --
cols = ["family", "task", "n_sizes", "n_front", "eps_inf", "alpha",
        "lambda_star", "boot_lo", "boot_hi", "boot_se", "R2", "S_range",
        "x_range", "d_aicc", "status"]
with open(f"{OUT}/table_ladders.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(table, key=lambda z: (z["status"] != "identified",
                                          z["family"], z["task"])):
        w.writerow(r)

json.dump(dict(
    n_candidate=cand, n_identified=len(ref), n_families=len(byfam),
    pooled=pool, by_family=stage1, two_stage=stage2,
    median_lam=float(np.median([ref[k]["lam"] for k in ref])) if ref else None,
    median_r2=float(np.median([ref[k]["r2"] for k in ref])) if ref else None,
    aicc_wins=int(sum(1 for v in aic.values() if v > 0)), aicc_n=len(aic),
    aicc_median=float(np.median([v for v in aic.values()])) if aic else None,
    n_below_half=sum(1 for k in ref if boot[k]["hi"] < .5),
    cells=[dict(label=f"{k[0]} / {k[1]}", family=k[0], task=k[1],
                lam=ref[k]["lam"], lo=boot[k]["lo"], hi=boot[k]["hi"],
                se=boot[k]["se"], R2=ref[k]["r2"], n_front=len(ref[k]["P"]),
                eps_inf=ref[k]["eps_inf"], d_aicc=aic.get(k),
                alpha=ref[k]["alpha"], c=ref[k]["c"],
                points=[[float(a), float(b)] for a, b in ref[k]["P"]])
           for k in sorted(ref, key=lambda z: ref[z]["lam"])],
    power_law=[dict(family=r["family"], task=r["task"],
                    n_front=r["n_front"], R2=r.get("R2"))
               for r in table
               if r["status"].startswith("excluded: floor")],
    vintage=vintage,
    vintage_median_rho=float(np.median([v["rho"] for v in vintage]))
    if vintage else None,
), open(f"{OUT}/stats_ladders.json", "w"), indent=1)
print(f"\nwrote {OUT}/table_ladders.csv, {OUT}/stats_ladders.json")
