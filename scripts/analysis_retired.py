#!/usr/bin/env python3
"""
analysis_retired.py -- the positive control.

Thirteen capacity ladders returning eps_inf = 0 is a failure to reject, not a
demonstration.  A demonstration needs a benchmark on which language models are
known, on external grounds, to have reached their ceiling, and it needs the
identifiability criterion to come out the other way there.

The design is a dissociation across benchmarks.  Six benchmarks of the retired
suite are scored on the same model suites, by the same harness, at the same
shot counts, so the benchmark is the only thing that varies.  Write eps_min for
the error of the best rung of a ladder; it is an endpoint of the data, fixed
before the saturating law is fitted.  The theory predicts, from Theorem 6
alone:

  P1  identification tracks how far the front has descended, not the domain;
  P2  where a human ceiling is published, the fitted floor respects it -- a
      constraint on a quantity that never enters the objective;
  P3  censoring a saturated ladder from the top drives eps_inf to zero, and
      censoring an ImageNet cell the same way does the same thing, so the
      language-model failure is reproducible on data where the floor is known
      to exist.

This script fits every ladder and records the fits.  The inference that tests
P1--P3 is in retired_stats.py, which reads what is written here.

Reads   corpus_retired.csv
Writes  table_retired.csv, table_retired_tokens.csv, stats_retired.json
"""
import collections
import csv
import json
import math
import os
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2, t as tdist, spearmanr, fisher_exact, mannwhitneyu

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260824)

DATA = os.environ.get("DATA", "data")
OUT = os.environ.get("OUT", "results")
os.makedirs(OUT, exist_ok=True)
B = int(os.environ.get("BOOT", 400))
B_TOK = int(os.environ.get("BOOT_TOK", 150))
MINFRONT = 6



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
    return (p[0], math.exp(p[1]), p[2], np.nan)


def fit_power(t, e):
    lr, ee = np.log(np.asarray(t, float)), np.asarray(e, float)
    f = lambda lr, lc, a: np.exp(lc) * np.exp(-a * lr)
    try:
        p, _ = curve_fit(f, lr, ee, p0=[0., .5], maxfev=400000,
                         bounds=([-60, .0], [60, 8]))
    except Exception:
        return None
    return math.exp(p[0]), p[1], float(((ee - f(lr, *p)) ** 2).sum())


def rss_knee(t, e, par):
    lr, ee = np.log(np.asarray(t, float)), np.asarray(e, float)
    return float(((ee - (par[0] + par[1] * np.exp(-par[2] * lr))) ** 2).sum())


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


def analyse(front, label, meta, boot=True, nboot=None):
    """Fit one ladder; return a row dict following the paper's rule."""
    row = dict(label=label, **meta, n_front=len(front))
    if len(front) < MINFRONT:
        row["status"] = "excluded: front shorter than six points"
        return row, None
    t = np.array([p[0] for p in front], float)
    e = np.array([p[1] for p in front], float)
    row["S_range"] = round(math.log(e.max() / e.min()), 3)
    row["x_range"] = round(math.log(t.max() / t.min()), 3)
    row["eps_min"] = round(float(e.min()), 6)
    f = fit_knee(t, e)
    if f is None or not (0 < f[2] < 8):
        row["status"] = "excluded: fit failed"
        return row, None
    ei, c, a, r2 = f
    row.update(eps_inf=round(ei, 6), alpha=round(a, 4),
               lambda_star=round(a / 2, 4), R2=round(r2, 4))
    pw = fit_power(t, e)
    if pw is not None:
        row["d_aicc"] = round(float(aicc(pw[2], len(t), 2)
                                    - aicc(rss_knee(t, e, (ei, c, a)), len(t), 3)), 2)
        row["R2_power"] = round(float(1 - pw[2] / max(((e - e.mean()) ** 2).sum(), 1e-300)), 4)
    if ei <= 1e-6 or ei >= .999 * e.min():
        row["status"] = "excluded: floor at boundary (degenerate)"
        return row, None
    row["status"] = "identified"
    rec = dict(front=front, eps_inf=ei, c=c, alpha=a, lam=a / 2, r2=r2)
    if boot:
        nb = nboot or B
        lams, floors, tries = [], [], 0
        while len(lams) < nb and tries < 4 * nb:
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
            floors.append(g[0])
        if len(lams) >= 30:
            lams, floors = np.array(lams), np.array(floors)
            rec["se"] = float(lams.std(ddof=1))
            rec["lo"], rec["hi"] = (float(np.percentile(lams, 2.5)),
                                    float(np.percentile(lams, 97.5)))
            rec["flo"], rec["fhi"] = (float(np.percentile(floors, 2.5)),
                                      float(np.percentile(floors, 97.5)))
        else:
            rec["se"], rec["lo"], rec["hi"] = a / 4, a / 4, a
            rec["flo"], rec["fhi"] = ei / 2, ei * 2
        row.update(boot_lo=round(rec["lo"], 4), boot_hi=round(rec["hi"], 4),
                   boot_se=round(rec["se"], 4),
                   floor_lo=round(rec["flo"], 6), floor_hi=round(rec["fhi"], 6))
    return row, rec


# ------------------------------------------------------------------ inputs --
rows = list(csv.DictReader(open(f"{DATA}/corpus_retired.csv")))
for r in rows:
    r["acc"] = float(r["acc"])
    r["chance"] = float(r["chance"])
    r["shots"] = int(r["shots"])
    r["step"] = int(r["step"] or 0)
    r["params"] = float(r["params"] or 0)
    r["eps"] = 1 - (r["acc"] - r["chance"]) / (1 - r["chance"])

FINAL = 143000
final = [r for r in rows if (r["step"] == FINAL or r["suite"] == "BLOOM")]

# ======================================================== A. capacity ladders
print("A  capacity ladders on the retired suite\n")
cap_table, cap_ref = [], {}
groups = collections.defaultdict(list)
for r in final:
    groups[(r["suite"], r["shots"], r["task"])].append(r)

for (suite, shots, task), v in sorted(groups.items()):
    by_p = {}
    for r in v:
        if r["params"] <= 0:
            continue
        if r["params"] not in by_p or r["eps"] < by_p[r["params"]]:
            by_p[r["params"]] = r["eps"]
    pts = [(p, e) for p, e in by_p.items() if 0 < e < 1]
    label = f"{suite} / {task}" + (f" / {shots}-shot" if shots else "")
    front = pareto(pts)
    meta = dict(suite=suite, task=task, shots=shots, resource="parameters",
                n_sizes=len(pts))
    row, rec = analyse(front, label, meta)
    cap_table.append(row)
    if rec:
        cap_ref[label] = dict(rec, **meta)
    st = row["status"].split(":")[0]
    print(f"  {label:44s} |P|={row['n_front']:2d} "
          f"{st:12s} eps_inf={row.get('eps_inf', float('nan')):.4f} "
          f"lam*={row.get('lambda_star', float('nan')):.3f} "
          f"R2={row.get('R2', float('nan')):.3f} "
          f"dAICc={row.get('d_aicc', float('nan')):+.1f}")

# ================================================== B. training-token ladders
print("\nB  training-token ladders (Pythia checkpoints, resource = tokens)\n")
tok_table, tok_ref = [], {}
tg = collections.defaultdict(list)
for r in rows:
    if not r["train_tokens"] or r["suite"].startswith(("OPT", "BLOOM")):
        continue
    tg[(r["suite"], r["model"], r["shots"], r["task"])].append(r)

for (suite, model, shots, task), v in sorted(tg.items()):
    pts = [(float(r["train_tokens"]), r["eps"]) for r in v
           if float(r["train_tokens"]) > 0 and 0 < r["eps"] < 1]
    if len(pts) < MINFRONT:
        continue
    front = pareto(pts)
    label = f"{suite} {model} / {task}" + (f" / {shots}-shot" if shots else "")
    meta = dict(suite=suite, model=model, task=task, shots=shots,
                resource="training tokens", n_sizes=len(pts))
    row, rec = analyse(front, label, meta, nboot=B_TOK)
    tok_table.append(row)
    if rec:
        tok_ref[label] = dict(rec, **meta)

ident_tok = [r for r in tok_table if r["status"] == "identified"]
print(f"  {len(ident_tok)} identified of "
      f"{len([r for r in tok_table if r['n_front'] >= MINFRONT])} candidates; "
      f"median |P| = {np.median([r['n_front'] for r in tok_table]):.0f}")
for r in sorted(ident_tok, key=lambda z: z["label"])[:12]:
    print(f"    {r['label']:46s} |P|={r['n_front']:2d} "
          f"eps_inf={r['eps_inf']:.4f} lam*={r['lambda_star']:.3f} "
          f"R2={r['R2']:.3f} dAICc={r.get('d_aicc', float('nan')):+.1f}")

# ------------------------------------------------------------------ outputs --
cols = ["label", "suite", "model", "task", "shots", "resource", "n_sizes",
        "n_front", "eps_min", "eps_inf", "floor_lo",
        "floor_hi", "alpha", "lambda_star", "boot_lo", "boot_hi", "boot_se",
        "R2", "R2_power", "S_range", "x_range", "d_aicc", "status"]
for fn, tb in (("table_retired.csv", cap_table),
               ("table_retired_tokens.csv", tok_table)):
    with open(f"{OUT}/{fn}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(tb, key=lambda z: (z["status"] != "identified", z["label"])):
            w.writerow(r)

json.dump(dict(
    n_records=len(rows),
    suites=sorted({r["suite"] for r in rows}),
    tasks=sorted({r["task"] for r in rows}),
    capacity=dict(
        n_candidate=len([r for r in cap_table if r["n_front"] >= MINFRONT]),
        n_identified=len(cap_ref),
        rows=cap_table),
    tokens=dict(
        n_candidate=len([r for r in tok_table if r["n_front"] >= MINFRONT]),
        n_identified=len(tok_ref),
        median_front=float(np.median([r["n_front"] for r in tok_table]))
        if tok_table else None),
    cells=[dict(label=l, task=d["task"], suite=d["suite"],
                resource=d["resource"], lam=d["lam"], lo=d.get("lo"),
                hi=d.get("hi"), se=d.get("se"), R2=d["r2"],
                eps_inf=d["eps_inf"], floor_lo=d.get("flo"),
                floor_hi=d.get("fhi"),
                n_front=len(d["front"]),
                points=[[float(a), float(b)] for a, b in d["front"]])
           for l, d in sorted({**cap_ref, **tok_ref}.items())],
), open(f"{OUT}/stats_retired.json", "w"), indent=1, default=float)
print(f"\nwrote {OUT}/table_retired.csv, {OUT}/table_retired_tokens.csv, "
      f"{OUT}/stats_retired.json")
