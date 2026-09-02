#!/usr/bin/env python3
"""
stats_llm.py  --  fit, audit and pool the canonical price on the task-resolved
language-model corpus produced by build_llm_corpus.py.

Applies exactly the machinery of stats_extended.py -- Pareto front, the
pre-declared six-point identifiability rule, the saturating fit, the
nonparametric bootstrap of lambda*, DerSimonian-Laird pooling, AICc against
the pure power law -- and adds three things the LLM evidence needs:

  L1  the information span S_max - S_min of every front, which is the
      quantity Section 12.3 identified as the reason the aggregate-metric
      cells failed, now reported for the task-resolved cells;
  L2  a two-stage pooling that respects the clustering of the corpus.  A
      platform x task design shares the same quality vector across platforms
      and the same model population across tasks, so the cells are not
      independent.  Stage one pools the platforms within a task, stage two
      pools the tasks; the result is one estimate per task and one overall,
      with as many independent units as there are tasks;
  L3  a front-length sweep, because the audit of Section 11 showed the
      three-parameter fit is only trustworthy on long fronts.

Writes table_llm.csv, stats_llm.json.
"""
import collections
import csv
import json
import math
import os
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2, t as tdist, norm, spearmanr

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260824)

DATA = os.environ.get("DATA", "data")
OUT = os.environ.get("OUT", "results")
os.makedirs(OUT, exist_ok=True)
B = int(os.environ.get("BOOT", 400))
MINFRONT = 6


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
    """DerSimonian-Laird random-effects pooling."""
    y = np.asarray(y, float)
    s2 = np.maximum(np.asarray(s2, float), 1e-12)
    k = len(y)
    if k == 0:
        return None
    if k == 1:
        se = math.sqrt(s2[0])
        return dict(k=1, mu=float(y[0]), se=se, tau=0.0, I2=0.0, Q=0.0,
                    Q_p=1.0, ci=[float(y[0] - 1.96 * se), float(y[0] + 1.96 * se)],
                    pi=[float(y[0]), float(y[0])])
    w = 1 / s2
    mufe = (w * y).sum() / w.sum()
    Q = float((w * (y - mufe) ** 2).sum())
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C)
    wr = 1 / (s2 + tau2)
    mu = float((wr * y).sum() / wr.sum())
    se = float(math.sqrt(1 / wr.sum()))
    tc = float(tdist.ppf(.975, max(k - 2, 1)))
    return dict(k=k, mu=mu, se=se, tau=math.sqrt(tau2), Q=Q,
                Q_p=float(1 - chi2.cdf(Q, k - 1)),
                I2=max(0.0, 100 * (Q - (k - 1)) / Q) if Q > 0 else 0.0,
                ci=[mu - 1.96 * se, mu + 1.96 * se],
                pi=[mu - tc * math.sqrt(tau2 + se ** 2),
                    mu + tc * math.sqrt(tau2 + se ** 2)])


# ------------------------------------------------------------------- cells --
cells, meta = collections.defaultdict(list), {}
for x in csv.DictReader(open(f"{DATA}/corpus_llm.csv")):
    Q, ch, t = float(x["Q"]), float(x["chance"]), float(x["latency_ms"])
    if t <= 0:
        continue
    e = 1 - (Q - ch) / (1 - ch)
    if 0 < e < 1:
        cells[x["cell"]].append((t, e))
        meta[x["cell"]] = (x["task"], x["source"], x["metric"])

table, ref = [], {}
for k, v in sorted(cells.items()):
    task, src, metric = meta[k]
    P = pareto(v)
    row = dict(cell=k, task=task, source=src, metric=metric,
               n=len(v), n_front=len(P))
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
    ref[k] = dict(P=P, eps_inf=ei, c=c, alpha=a, r2=r2, lam=a / 2,
                  task=task, source=src, S=row.get("S_range", np.nan))

print(f"candidate cells {len(cells)}   fronts >= {MINFRONT}: "
      f"{sum(1 for r in table if 'lambda_star' in r or r['status'].startswith('excluded: floor'))}"
      f"   identified {len(ref)}")


# --------------------------------------------------------------- bootstrap --
boot = {}
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
                   hi=float(np.percentile(lams, 97.5)), n=len(lams))
    print(f"  {k:56s} lam*={d['lam']:.3f} "
          f"[{boot[k]['lo']:.3f},{boot[k]['hi']:.3f}] |P|={len(P)} "
          f"S={d['S']:.2f} R2={d['r2']:.3f}", flush=True)

for r in table:
    if r["cell"] in boot:
        b = boot[r["cell"]]
        r.update(boot_lo=round(b["lo"], 4), boot_hi=round(b["hi"], 4),
                 boot_se=round(b["se"], 4))


# ------------------------------------------------------------------- AICc ---
aic = {}
for k, d in ref.items():
    P = np.array(d["P"])
    t, e, n = P[:, 0], P[:, 1], len(P)
    pw = fit_power(t, e)
    if pw is None:
        continue
    aic[k] = float(aicc(pw[2], n, 2) -
                   aicc(rss_knee(t, e, (d["eps_inf"], d["c"], d["alpha"])), n, 3))
for r in table:
    if r["cell"] in aic:
        r["d_aicc"] = round(aic[r["cell"]], 2)


# --------------------------------------------------------- two-stage pool ---
def pool_keys(keys):
    return dl([ref[k]["lam"] for k in keys], [boot[k]["se"] ** 2 for k in keys])


by_task = collections.defaultdict(list)
for k in ref:
    by_task[ref[k]["task"]].append(k)

stage1 = {t: pool_keys(ks) for t, ks in sorted(by_task.items())}
stage2 = dl([p["mu"] for p in stage1.values()],
            [p["se"] ** 2 for p in stage1.values()])
flat = pool_keys(sorted(ref))

print(f"\nstage 1: {len(stage1)} tasks")
for t, p in sorted(stage1.items(), key=lambda z: z[1]["mu"]):
    print(f"   {t:26s} k={p['k']:2d}  {p['mu']:.3f} "
          f"[{p['ci'][0]:.3f},{p['ci'][1]:.3f}]")
print(f"stage 2 (task-level, k={stage2['k']}): {stage2['mu']:.3f} "
      f"[{stage2['ci'][0]:.3f},{stage2['ci'][1]:.3f}] tau={stage2['tau']:.3f} "
      f"I2={stage2['I2']:.0f}%")
print(f"flat pool over {flat['k']} cells:  {flat['mu']:.3f} "
      f"[{flat['ci'][0]:.3f},{flat['ci'][1]:.3f}] tau={flat['tau']:.3f}")


# ------------------------------------------------- front-length restriction --
sweep = {}
for mf in (6, 8, 10, 12):
    ks = [k for k in ref if len(ref[k]["P"]) >= mf]
    if len(ks) < 2:
        continue
    p = pool_keys(ks)
    sweep[mf] = dict(k=len(ks), mu=p["mu"], ci=p["ci"], tau=p["tau"],
                     median=float(np.median([ref[k]["lam"] for k in ks])))
    print(f"   front >= {mf:2d}: k={len(ks):2d} pooled {p['mu']:.3f} "
          f"[{p['ci'][0]:.3f},{p['ci'][1]:.3f}] "
          f"median {sweep[mf]['median']:.3f}")

# the informative subset: a front long enough to fit and wide enough to see
SUB = [k for k in ref if len(ref[k]["P"]) >= 8 and ref[k]["S"] >= 0.5]
sub = pool_keys(SUB) if len(SUB) >= 2 else None
if sub:
    print(f"\ninformative subset (|P|>=8 and S>=0.5): k={sub['k']} "
          f"pooled {sub['mu']:.3f} [{sub['ci'][0]:.3f},{sub['ci'][1]:.3f}] "
          f"tau={sub['tau']:.3f} I2={sub['I2']:.0f}% "
          f"PI [{sub['pi'][0]:.3f},{sub['pi'][1]:.3f}]")


# --------------------------------------------------------- span diagnostic --
have = [r for r in table if "S_range" in r and r["n_front"] >= MINFRONT]
S_id = [r["S_range"] for r in have if r["status"] == "identified"]
S_no = [r["S_range"] for r in have if r["status"] != "identified"]
rho = spearmanr([r["S_range"] for r in have],
                [1 if r["status"] == "identified" else 0 for r in have])
print(f"\ninformation span: identified cells median {np.median(S_id):.2f} nats, "
      f"not identified {np.median(S_no):.2f} nats "
      f"(Spearman rho={rho.statistic:.2f}, p={rho.pvalue:.1e})")
allS = [r["S_range"] for r in table if "S_range" in r]
print(f"median span over all {len(allS)} candidate fronts: {np.median(allS):.2f} nats")

# how the same platforms scored under the aggregate metric, for contrast
AGG = 0.28

below = [k for k in ref if boot[k]["hi"] < .5]
above = [k for k in ref if boot[k]["lo"] > .5]
print(f"bound lambda* < 1/2: {len(below)} of {len(ref)} cells entirely below, "
      f"{len(above)} entirely above")

da = np.array(list(aic.values()))
print(f"AICc: saturating preferred in {(da > 0).sum()}/{len(da)}, "
      f"median {np.median(da):.1f}")


# ----------------------------------------------------------------- outputs --
cols = ["cell", "task", "source", "metric", "n", "n_front", "eps_inf", "alpha",
        "lambda_star", "boot_lo", "boot_hi", "boot_se", "R2", "S_range",
        "x_range", "d_aicc", "status"]
with open(f"{OUT}/table_llm.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(table, key=lambda z: (z["status"] != "identified", z["cell"])):
        w.writerow(r)

json.dump(dict(
    n_records=sum(len(v) for v in cells.values()),
    n_candidate=len(cells),
    n_front_ok=sum(1 for r in table if r["n_front"] >= MINFRONT),
    n_identified=len(ref),
    stage1=stage1, stage2=stage2, flat=flat, sweep=sweep,
    subset=dict(k=len(SUB), pooled=sub, keys=sorted(SUB)),
    cells=[dict(label=k, task=ref[k]["task"], source=ref[k]["source"],
                lam=ref[k]["lam"], lo=boot[k]["lo"], hi=boot[k]["hi"],
                se=boot[k]["se"], eps_inf=ref[k]["eps_inf"],
                alpha=ref[k]["alpha"], R2=ref[k]["r2"],
                n_front=len(ref[k]["P"]), S=ref[k]["S"], d_aicc=aic.get(k))
           for k in sorted(ref, key=lambda z: ref[z]["lam"])],
    span=dict(median_all=float(np.median(allS)),
              median_identified=float(np.median(S_id)),
              median_not_identified=float(np.median(S_no)),
              aggregate_reference=AGG,
              spearman_rho=float(rho.statistic), spearman_p=float(rho.pvalue)),
    n_below_half=len(below), n_above_half=len(above),
    aicc_wins=int((da > 0).sum()), aicc_n=int(len(da)),
    aicc_median=float(np.median(da)),
), open(f"{OUT}/stats_llm.json", "w"), indent=1)
print(f"\nwrote {OUT}/table_llm.csv, {OUT}/stats_llm.json")
