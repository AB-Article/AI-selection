#!/usr/bin/env python3
"""
floor_identification.py  --  is the irreducible error identified, or is the
fitted floor just the constraint boundary?

The estimator used throughout this paper minimises squared residuals in
epsilon and bounds the floor above by 0.999*min(eps). In seven of the nine
long-front cells the returned floor sits exactly on that bound, which means the
floor was not estimated at all. This script asks why, and whether a better
estimator recovers it.

F1  Profile the residual sum of squares over a grid of fixed floors, refitting
    (c, alpha) at each. If the profile has an interior minimum the floor is
    identified; if it decreases monotonically to the boundary it is not.

F2  Refit in log space. robustness.py established that the quality noise is
    multiplicative with sigma ~ 0.13, so least squares in epsilon is the wrong
    criterion: it weights the large-error, small-latency end of the front by
    orders of magnitude more than the small-error end where the floor lives.
    Least squares in log epsilon is the maximum-likelihood criterion under the
    noise model the calibration already assumes.

F3  Model comparison per cell by AICc, in log space: pure power law
    (floor = 0) against the saturating law with a free floor. This is the test
    of whether the data support a strictly positive floor at all -- the
    condition Theorem 11 requires for a price to exist.

F4  Recompute the price under the log-space estimator and repool.

Writes floor_identification.json, table_floor.csv
"""
import collections
import csv
import json
import math
import os
import warnings

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import chi2, f as fdist, t as tdist

warnings.filterwarnings("ignore")

_D = os.environ.get("DATA", "data")
_OUT = os.environ.get("OUT", "results")
D, OUT = _D, _OUT
os.makedirs(OUT, exist_ok=True)
MINFRONT = int(os.environ.get("MINFRONT", 20))


def pareto(pts):
    pts = sorted(pts)
    out, best = [], np.inf
    for t, e in pts:
        if e < best - 1e-15:
            out.append((t, e))
            best = e
    return out


# ------------------------------------------------------------ estimators ----
def fit_lin(t, e, floor_cap=0.999):
    """Least squares in epsilon -- the estimator used in the paper."""
    lr, ee = np.log(t), np.asarray(e, float)
    f = lambda lr, ei, lc, a: ei + np.exp(lc) * np.exp(-a * lr)
    hi = ee.min() * floor_cap
    best = None
    for s in (0., .3, .6, .9):
        try:
            p, _ = curve_fit(f, lr, ee, p0=[hi * s, 0., .5], maxfev=400000,
                             bounds=([0, -40, .01], [hi, 40, 8]))
        except Exception:
            continue
        rss = float(((ee - f(lr, *p)) ** 2).sum())
        if best is None or rss < best[3]:
            best = (p[0], math.exp(p[1]), p[2], rss)
    return best


def sse_log(params, lr, le):
    ei, lc, a = params
    m = ei + np.exp(lc) * np.exp(-a * lr)
    if np.any(m <= 0) or not np.all(np.isfinite(m)):
        return 1e30
    return float(((le - np.log(m)) ** 2).sum())


def _sig(z, scale=1.0):
    z = max(-500.0, min(500.0, float(z)))
    return scale / (1.0 + math.exp(-z))


def _unpack(u, hi):
    """Map R^3 to (floor in [0,hi], log c, alpha in (0,8))."""
    return _sig(u[0], hi), u[1], _sig(u[2], 8.0)


def fit_log(t, e, floor_cap=0.99999):
    """Least squares in log epsilon -- ML under multiplicative noise."""
    lr, le = np.log(t), np.log(np.asarray(e, float))
    hi = float(np.exp(le).min()) * floor_cap
    obj = lambda u: sse_log(list(_unpack(u, hi)), lr, le)
    best = None
    for s0 in (-6., -2., 0., 2., 6.):
        for a0 in (-2.7, -1.6, 0.0):
            r = minimize(obj, [s0, 0., a0], method="Nelder-Mead",
                         options=dict(maxiter=40000, maxfev=40000,
                                      xatol=1e-11, fatol=1e-13))
            ei, lc, a = _unpack(r.x, hi)
            if not np.isfinite(r.fun) or r.fun > 1e29:
                continue
            if best is None or r.fun < best[3]:
                best = (ei, math.exp(lc), a, float(r.fun))
    return best


def fit_log_fixed_floor(t, e, ei):
    """(c, alpha) at a fixed floor, in log space; returns SSE."""
    lr, le = np.log(t), np.log(np.asarray(e, float))
    g = lambda p: sse_log([ei, p[0], _sig(p[1], 8.0)], lr, le)
    best = None
    for a0 in (-2.7, -1.6, 0.0, 1.0):
        r = minimize(g, [0., a0], method="Nelder-Mead",
                     options=dict(maxiter=40000, maxfev=40000,
                                  xatol=1e-11, fatol=1e-13))
        if best is None or r.fun < best[1]:
            best = (r.x, float(r.fun))
    return (math.exp(best[0][0]), _sig(best[0][1], 8.0), best[1])


def aicc(sse, n, k):
    return np.nan if n - k - 1 <= 0 else \
        n * math.log(sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


# ------------------------------------------------------------------ cells ---
cells = collections.defaultdict(list)
for x in csv.DictReader(open(f"{D}/corpus.csv")):
    Q, ch, t = float(x["Q"]), float(x["chance"]), float(x["latency_ms"])
    if t <= 0:
        continue
    e = 1 - (Q - ch) / (1 - ch)
    if 0 < e < 1:
        cells[(x["source"], x["cell"])].append((t, e))
if os.path.exists(f"{D}/corpus_extended.csv"):
    for x in csv.DictReader(open(f"{D}/corpus_extended.csv")):
        Q, ch, t = float(x["Q"]), float(x["chance"]), float(x["latency_ms"])
        if t <= 0:
            continue
        e = 1 - (Q - ch) / (1 - ch)
        if 0 < e < 1:
            cells[(x["source"], x["cell"])].append((t, e))

fronts = {k: pareto(v) for k, v in cells.items()}
fronts = {k: P for k, P in fronts.items() if len(P) >= MINFRONT}
print(f"cells with fronts >= {MINFRONT} points: {len(fronts)}\n")

rows = []
for k, P in sorted(fronts.items()):
    t = np.array([p[0] for p in P])
    e = np.array([p[1] for p in P])
    n = len(P)
    emin = float(e.min())
    name = str(k[1])[:46]

    lin = fit_lin(t, e)
    log = fit_log(t, e)
    if lin is None or log is None:
        continue

    # ---- F1 profile over the floor, in log space
    grid = np.concatenate([[0.0], np.linspace(1e-4, emin * 0.99999, 160)])
    sse = np.array([fit_log_fixed_floor(t, e, g)[2] for g in grid])
    j = int(np.argmin(sse))
    interior = 0 < j < len(grid) - 1
    sse_min = float(sse[j])
    # profile-likelihood interval: F-test threshold on one parameter
    thr = sse_min * (1 + fdist.ppf(0.95, 1, n - 3) / (n - 3))
    ok = grid[sse <= thr]
    lo_f, hi_f = float(ok.min()), float(ok.max())
    floor_identified = interior and hi_f < emin * 0.99 and lo_f > 1e-6

    # ---- F3 model comparison in log space: floor = 0 vs floor free
    _, a0, sse0 = fit_log_fixed_floor(t, e, 0.0)
    d_aicc = aicc(sse0, n, 2) - aicc(sse_min, n, 3)

    rows.append(dict(
        cell=name, n_front=n, eps_min=emin,
        floor_lin=lin[0], ratio_lin=lin[0] / emin, lam_lin=lin[2] / 2,
        floor_log=log[0], ratio_log=log[0] / emin, lam_log=log[2] / 2,
        floor_profile=float(grid[j]), profile_lo=lo_f, profile_hi=hi_f,
        interior_minimum=bool(interior),
        floor_identified=bool(floor_identified),
        d_aicc_floor=float(d_aicc), alpha_powerlaw=float(a0)))
    print(f"{name:46s} |P|={n:3d}")
    print(f"    eps-space fit : floor {lin[0]:.4f} = {100*lin[0]/emin:5.1f}% of "
          f"min eps   lambda* {lin[2]/2:.3f}")
    print(f"    log-space fit : floor {log[0]:.4f} = {100*log[0]/emin:5.1f}% of "
          f"min eps   lambda* {log[2]/2:.3f}")
    print(f"    profile       : floor {grid[j]:.4f}  95% CI "
          f"[{lo_f:.4f},{hi_f:.4f}]  interior={interior}  "
          f"identified={floor_identified}")
    print(f"    floor>0 vs floor=0 : dAICc {d_aicc:+.1f}\n")

# ------------------------------------------------------------------ summary --
nid = sum(r["floor_identified"] for r in rows)
pin = sum(r["ratio_lin"] > 0.99 for r in rows)
print("=" * 72)
print(f"floors pinned to the bound under the eps-space estimator : {pin}/{len(rows)}")
print(f"floors pinned to the bound under the log-space estimator : "
      f"{sum(r['ratio_log'] > 0.99 for r in rows)}/{len(rows)}")
print(f"floors identified by profile likelihood                  : {nid}/{len(rows)}")
da = np.array([r["d_aicc_floor"] for r in rows])
print(f"AICc prefers a positive floor in                          : "
      f"{(da > 0).sum()}/{len(rows)} cells (median {np.median(da):+.1f})")

lam_lin = np.array([r["lam_lin"] for r in rows])
lam_log = np.array([r["lam_log"] for r in rows])
print(f"\nlambda*  eps-space : median {np.median(lam_lin):.3f}  "
      f"range [{lam_lin.min():.3f},{lam_lin.max():.3f}]")
print(f"lambda*  log-space : median {np.median(lam_log):.3f}  "
      f"range [{lam_log.min():.3f},{lam_log.max():.3f}]")
print(f"median shift: {np.median(lam_log) - np.median(lam_lin):+.3f}")

sub = [r for r in rows if r["floor_identified"]]
if sub:
    ll = np.array([r["lam_log"] for r in sub])
    print(f"\nover the {len(sub)} cells with an identified floor: "
          f"median lambda* {np.median(ll):.3f}, "
          f"range [{ll.min():.3f},{ll.max():.3f}], "
          f"sd {ll.std(ddof=1) if len(ll) > 1 else float('nan'):.3f}")

with open(f"{OUT}/table_floor.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow(r)
json.dump(dict(cells=rows, minfront=MINFRONT,
               n_pinned_lin=int(pin),
               n_pinned_log=int(sum(r["ratio_log"] > 0.99 for r in rows)),
               n_identified=int(nid),
               n_aicc_floor=int((da > 0).sum()),
               median_lam_lin=float(np.median(lam_lin)),
               median_lam_log=float(np.median(lam_log))),
          open(f"{OUT}/floor_identification.json", "w"), indent=1)
print("\nwrote floor_identification.json, table_floor.csv")
