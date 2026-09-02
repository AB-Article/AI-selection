#!/usr/bin/env python3
"""
decision_logspace.py  --  re-score the two-system decision experiment against
frontiers fitted by the final estimator.

The agreement figures in the results section were produced by analysis_main.py,
which fits each cell's frontier by least squares in epsilon. finalize.py
established that least squares in log epsilon is the right criterion for the
multiplicative scatter the residual diagnostics confirm, and it is the estimator
behind the reported price. The decision experiment was never re-run under it, so
the ground truth it scores against came from one estimator while the headline
number came from another.

This script closes that gap. The pair construction, the separation threshold,
the rules and the scoring are identical to analysis_main.py; only the frontier
fit changes. Both sets of numbers are reported side by side.

Writes decision_logspace.json, table_rules_logspace.csv
"""
import collections
import csv
import itertools
import json
import math
import os
import warnings

import numpy as np
from scipy.optimize import curve_fit, minimize

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260824)

_D = os.environ.get("DATA", "data")
_OUT = os.environ.get("OUT", "results")
D, OUT = _D, _OUT
os.makedirs(OUT, exist_ok=True)

MINFRONT = 6
SEP = math.log(1.5)


def pareto(pts):
    pts = sorted(pts)
    out, best = [], np.inf
    for t, e in pts:
        if e < best - 1e-15:
            out.append((t, e))
            best = e
    return out


def fit_eps(t, e):
    """Least squares in epsilon -- the estimator of analysis_main.py."""
    lr, ee = np.log(t), np.array(e)
    f = lambda lr, ei, lc, a: ei + np.exp(lc) * np.exp(-a * lr)
    best = None
    for s in (0., .3, .6, .9):
        try:
            p, _ = curve_fit(f, lr, ee, p0=[min(ee) * s, 0., .5], maxfev=400000,
                             bounds=([0, -40, .01], [min(ee) * .999, 40, 8]))
        except Exception:
            continue
        r = ee - f(lr, *p)
        ss = 1 - (r ** 2).sum() / ((ee - ee.mean()) ** 2).sum()
        if best is None or ss > best[3]:
            best = (p[0], math.exp(p[1]), p[2], ss)
    return best


def _sig(z, s):
    z = max(-500.0, min(500.0, float(z)))
    return s / (1.0 + math.exp(-z))


def _sse(u, lr, le, hi):
    ei, lc, a = _sig(u[0], hi), u[1], _sig(u[2], 8.0)
    m = ei + math.exp(lc) * np.exp(-a * lr)
    if np.any(m <= 0) or not np.all(np.isfinite(m)):
        return 1e30
    return float(((le - np.log(m)) ** 2).sum())


def fit_log(t, e):
    """Least squares in log epsilon -- the estimator of finalize.py.

    Dense multi-start: the objective has a spurious local optimum at floor = 0
    whose SSE can be several times the global one (see CHANGES, ninth revision).
    """
    lr, le = np.log(np.asarray(t, float)), np.log(np.asarray(e, float))
    hi = float(np.exp(le).min())
    best = None
    for s0 in (-8., -6., -4., -2., -1., 0., 1., 3., 6., 8.):
        for a0 in (-3.5, -2.7, -1.6, -0.8, 0., 1.):
            r = minimize(_sse, [s0, 0., a0], args=(lr, le, hi),
                         method="Nelder-Mead",
                         options=dict(maxiter=30000, maxfev=30000,
                                      xatol=1e-11, fatol=1e-13))
            if r.fun < 1e29 and (best is None or r.fun < best[3]):
                best = (_sig(r.x[0], hi), math.exp(r.x[1]),
                        _sig(r.x[2], 8.0), float(r.fun))
    if best is None:
        return None
    m = best[0] + best[1] * np.exp(-best[2] * lr)
    r2 = 1 - float(((le - np.log(m)) ** 2).sum()) / \
        float(((le - le.mean()) ** 2).sum())
    return (best[0], best[1], best[2], r2)


# ------------------------------------------------------------------ cells ---
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


def build(fitter, require_interior_floor=True):
    """require_interior_floor: the exclusion used by analysis_main.py.

    It is appropriate for the epsilon-space estimator. It is NOT appropriate for
    the log-space one: floor_identification.py showed the profile likelihood for
    the floor is monotone to the boundary in nine of nine cells, so a floor at
    min(eps) is the correct one-sided estimate rather than a failure, and the
    price is insensitive to where in its profile interval the floor sits
    (median width 0.013). Applying the interior-floor rule to the log-space fits
    would discard ten of the twelve cells for passing the test the paper itself
    says they should pass.
    """
    ref = {}
    for k, v in cells.items():
        P = pareto(v)
        if len(P) < MINFRONT:
            continue
        f = fitter([p[0] for p in P], [p[1] for p in P])
        if f is None:
            continue
        ei, c, a, r2 = f
        if not (0 < a < 8):
            continue
        if ei <= 1e-6:
            continue
        if require_interior_floor and ei >= 0.999 * min(p[1] for p in P):
            continue
        ref[k] = dict(P=P, eps_inf=ei, c=c, alpha=a, lam=a / 2, r2=r2)
    return ref


def experiment(ref, tag):
    lams = np.array([d["lam"] for d in ref.values()])
    lam_hat = float(np.median(lams))
    pairs = []
    for k, d in ref.items():
        for (tf, ef), (ts, es) in itertools.combinations(d["P"], 2):
            if ts <= tf or es >= ef:
                continue
            dlr = math.log(ts / tf)
            if dlr < SEP:
                continue
            lab = math.log(ef / es) / dlr
            pairs.append(dict(k=k, dlr=dlr, lab=lab,
                              truth="slow" if lab > d["lam"] else "fast"))
    loco = {k: float(np.median([d["lam"] for kk, d in ref.items() if kk != k]))
            for k in ref}
    loto = {}
    for k in ref:
        o = [d["lam"] for kk, d in ref.items() if kk[2] != k[2]]
        loto[k] = float(np.median(o)) if o else lam_hat

    def score(fn):
        ok = und = 0
        reg = 0.0
        for p in pairs:
            v = fn(p)
            if v is None:
                und += 1
                continue
            if v == p["truth"]:
                ok += 1
            else:
                reg += abs(ref[p["k"]]["lam"] - p["lab"]) * p["dlr"]
        dec = len(pairs) - und
        return dict(acc=100 * ok / dec if dec else 0.0,
                    regret=reg / len(pairs))

    price = lambda L: (lambda p: "slow" if p["lab"] > L else "fast")
    rules = collections.OrderedDict()
    rules["Universal knee price $\\lambda=0.35$"] = price(0.35)
    rules["Universal knee price $\\lambda=0.319$"] = price(0.319)
    rules["Leave-one-cell-out knee price"] = \
        lambda p: "slow" if p["lab"] > loco[p["k"]] else "fast"
    rules["Leave-one-task-out knee price"] = \
        lambda p: "slow" if p["lab"] > loto[p["k"]] else "fast"
    for L in (0.166, 0.2, 0.25, 0.3, 0.4, 0.5):
        rules[f"Fixed $\\lambda={L}$"] = price(L)
    rules["Accuracy per unit time ($\\varepsilon t$)"] = price(1.0)
    rules["Always the faster system"] = lambda p: "fast"
    rules["Always the more accurate system"] = lambda p: "slow"
    rules["Coin flip"] = lambda p: "slow" if RNG.random() < .5 else "fast"

    out = {n: score(f) for n, f in rules.items()}
    V = np.array([p["truth"] == "slow" for p in pairs])
    W = np.array([p["lab"] > 0.35 for p in pairs])
    po = float((V == W).mean())
    pe = float(V.mean() * W.mean() + (1 - V.mean()) * (1 - W.mean()))
    kappa = (po - pe) / (1 - pe)
    sweep = {f"{L:.2f}": score(price(L))["acc"]
             for L in np.arange(0.05, 1.01, 0.05)}
    print(f"\n[{tag}]  cells={len(ref)}  pairs={len(pairs)}  "
          f"median lambda*={lam_hat:.3f}  kappa={kappa:.3f}")
    for n, r in out.items():
        print(f"   {n:46s} {r['acc']:6.2f}  regret {r['regret']:.4f}")
    return dict(n_cells=len(ref), n_pairs=len(pairs), lam_hat=lam_hat,
                kappa=kappa, rules=out, sweep=sweep)


ref_eps = build(fit_eps)
ref_log = build(fit_log, require_interior_floor=False)
res_eps = experiment(ref_eps, "epsilon-space fits, as published")
res_log = experiment(ref_log, "log-space fits, the final estimator")

k35 = "Universal knee price $\\lambda=0.35$"
k319 = "Universal knee price $\\lambda=0.319$"
d = res_log["rules"][k35]["acc"] - res_eps["rules"][k35]["acc"]
print("\n" + "=" * 72)
print(f"agreement at lambda=0.35 : {res_eps['rules'][k35]['acc']:.2f}% "
      f"(eps-space) -> {res_log['rules'][k35]['acc']:.2f}% (log-space), "
      f"{d:+.2f} points")
print(f"agreement at lambda=0.319: {res_log['rules'][k319]['acc']:.2f}% "
      f"under the log-space fits")
best = max(res_log["sweep"].items(), key=lambda z: z[1])
print(f"best declared price on the log-space sweep: lambda={best[0]} "
      f"at {best[1]:.2f}%")
print("=" * 72)

with open(f"{OUT}/table_rules_logspace.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["rule", "accuracy_pct_eps_space", "accuracy_pct_log_space",
                "regret_log_space"])
    for n in res_log["rules"]:
        a_e = res_eps["rules"].get(n, {}).get("acc", float("nan"))
        w.writerow([n, f"{a_e:.2f}", f"{res_log['rules'][n]['acc']:.2f}",
                    f"{res_log['rules'][n]['regret']:.4f}"])

json.dump(dict(eps_space=res_eps, log_space=res_log,
               delta_at_035=d,
               acc_035_log=res_log["rules"][k35]["acc"],
               acc_0319_log=res_log["rules"][k319]["acc"],
               best_price_log=best[0], best_acc_log=best[1]),
          open(f"{OUT}/decision_logspace.json", "w"), indent=1)
print("\nwrote decision_logspace.json, table_rules_logspace.csv")
