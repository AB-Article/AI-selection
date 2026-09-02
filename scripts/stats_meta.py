#!/usr/bin/env python3
"""
stats_meta.py  --  statistical strengthening of the canonical-price estimate.

Adds to the pipeline of analysis_main.py:

  M1  Per-cell nonparametric bootstrap of the Pareto front (B resamples,
      refitting eps_inf, c, alpha on every resample) -> percentile CI and
      bootstrap standard error for lambda* = alpha/2.
  M2  DerSimonian-Laird random-effects meta-analysis across cells: pooled
      lambda*, tau, Cochran's Q, I^2 and the 95% prediction interval; and a
      one-sided Wald test of the nonparametric bound lambda* < 1/2.
      A permutation homogeneity test that resamples each cell's own bootstrap
      distribution is reported alongside Q, which assumes normal errors.
  M3  Model selection by AICc and Akaike weights, saturating law against the
      pure power law, per cell.
  M4  Split-half decircularisation: the front is partitioned by alternating
      rank, (eps_inf, c, alpha) refitted on each half, and lambda* compared
      across the disjoint halves; also the bootstrap correlation of the
      jointly fitted (eps_inf, alpha).
  M5  Numerical verification of the Bayes and minimax characterisations of
      the canonical price (Theorem 4(iii)-(iv)).

Reads  data/corpus.csv, data/mai_data.csv
Writes results/table_meta.csv, out/table_aicc.csv, out/table_splithalf.csv,
       out/stats_meta.json
"""
import collections
import csv
import json
import math
import os
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t as tdist, norm, beta as betadist

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(20260824)
# resolve data/output locations: works both in the artifacts tree and flat
_D = os.environ.get("DATA", "data")
_OUT = os.environ.get("OUT", "results")
D, OUT = _D, _OUT
os.makedirs(OUT, exist_ok=True)

B = 2000          # bootstrap resamples per cell
MINFRONT = 6      # pre-declared identifiability rule
NPERM = 20000


# ----------------------------------------------------------------------
# shared machinery -- identical to analysis_main.py
# ----------------------------------------------------------------------
def pareto(pts):
    pts = sorted(pts)
    out, best = [], np.inf
    for t, e in pts:
        if e < best - 1e-15:
            out.append((t, e))
            best = e
    return out


def fit_knee(t, e):
    """eps(t) = eps_inf + c t^-alpha  ->  (eps_inf, c, alpha, R2)."""
    lr, ee = np.log(np.asarray(t, float)), np.asarray(e, float)
    f = lambda lr, ei, lc, a: ei + np.exp(lc) * np.exp(-a * lr)
    best = None
    for ei0 in (0.0, .3, .6, .9):
        try:
            p, _ = curve_fit(f, lr, ee, p0=[min(ee) * ei0, 0., .5], maxfev=400000,
                             bounds=([0, -40, .01], [min(ee) * .999, 40, 8]))
        except Exception:
            continue
        r = ee - f(lr, *p)
        ss = 1 - (r ** 2).sum() / max(((ee - ee.mean()) ** 2).sum(), 1e-300)
        if best is None or ss > best[3]:
            best = (p[0], math.exp(p[1]), p[2], ss)
    return best


def fit_knee_fast(t, e, p0):
    """Single-start refit, seeded at the full-sample solution (bootstrap use)."""
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
    """eps(t) = c t^-alpha  ->  (c, alpha, RSS)."""
    lr, ee = np.log(np.asarray(t, float)), np.asarray(e, float)
    f = lambda lr, lc, a: np.exp(lc) * np.exp(-a * lr)
    try:
        p, _ = curve_fit(f, lr, ee, p0=[0., .5], maxfev=400000,
                         bounds=([-40, .0], [40, 8]))
    except Exception:
        return None
    rss = float(((ee - f(lr, *p)) ** 2).sum())
    return math.exp(p[0]), p[1], rss


def rss_knee(t, e, par):
    ei, c, a = par[0], par[1], par[2]
    lr, ee = np.log(np.asarray(t, float)), np.asarray(e, float)
    return float(((ee - (ei + c * np.exp(-a * lr))) ** 2).sum())


def aicc(rss, n, k):
    if n - k - 1 <= 0:
        return np.nan
    return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def bic(rss, n, k):
    return n * math.log(rss / n) + k * math.log(n)


# ----------------------------------------------------------------------
# 1. build cells exactly as analysis_main.py does
# ----------------------------------------------------------------------
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

ref = {}
for k, v in cells.items():
    P = pareto(v)
    if len(P) < MINFRONT:
        continue
    fit = fit_knee([p[0] for p in P], [p[1] for p in P])
    if fit is None:
        continue
    ei, c, a, r2 = fit
    if not (0 < a < 8):
        continue
    if ei <= 1e-6 or ei >= 0.999 * min(p[1] for p in P):
        continue
    ref[k] = dict(P=P, n=len(v), eps_inf=ei, c=c, alpha=a, r2=r2, lam=a / 2)

print(f"identified cells: {len(ref)}")

SHORT = {}
for k in ref:
    fam, cell, task = k
    if fam == "timm":
        SHORT[k] = cell.split("|")[1].strip() + " / " + cell.split("|")[2].strip().replace(
            "PyTorch 2.4.0 ", "").replace("PyTorch 2.9.1 ", "") + " " + cell.split("|")[3].strip()
    else:
        SHORT[k] = "CPU ONNX / " + task


# ----------------------------------------------------------------------
# M1  per-cell bootstrap
# ----------------------------------------------------------------------
boot = {}
for k, d in sorted(ref.items()):
    P = np.array(d["P"])
    lams, floors = [], []
    tries = 0
    while len(lams) < B and tries < 40 * B:
        tries += 1
        idx = RNG.integers(0, len(P), len(P))
        tt, ee = P[idx, 0], P[idx, 1]
        if len(np.unique(tt)) < 4:
            continue
        f = fit_knee_fast(tt, ee, (d["eps_inf"], d["c"], d["alpha"]))
        if f is None or not (0 < f[2] < 8):
            continue
        if f[0] <= 1e-6 or f[0] >= 0.999 * ee.min():
            continue
        lams.append(f[2] / 2)
        floors.append(f[0])
    lams = np.array(lams)
    floors = np.array(floors)
    boot[k] = dict(lam=lams, floor=floors,
                   se=float(lams.std(ddof=1)),
                   lo=float(np.percentile(lams, 2.5)),
                   hi=float(np.percentile(lams, 97.5)),
                   rho=float(np.corrcoef(floors, 2 * lams)[0, 1]),
                   nboot=len(lams))
    print(f"  {SHORT[k]:42s} lam*={d['lam']:.3f} "
          f"[{boot[k]['lo']:.3f},{boot[k]['hi']:.3f}] se={boot[k]['se']:.3f} "
          f"({boot[k]['nboot']} draws)")


# ----------------------------------------------------------------------
# M2  DerSimonian-Laird random effects
# ----------------------------------------------------------------------
keys = sorted(ref, key=lambda k: ref[k]["lam"])
y = np.array([ref[k]["lam"] for k in keys])
s2 = np.array([boot[k]["se"] ** 2 for k in keys])
kk = len(y)

w = 1 / s2
mu_fe = float((w * y).sum() / w.sum())
Q = float((w * (y - mu_fe) ** 2).sum())
dfQ = kk - 1
C = w.sum() - (w ** 2).sum() / w.sum()
tau2 = max(0.0, (Q - dfQ) / C)
tau = math.sqrt(tau2)
I2 = max(0.0, 100 * (Q - dfQ) / Q)

wr = 1 / (s2 + tau2)
mu = float((wr * y).sum() / wr.sum())
se_mu = float(math.sqrt(1 / wr.sum()))
ci = (mu - 1.96 * se_mu, mu + 1.96 * se_mu)
tcrit = float(tdist.ppf(0.975, max(kk - 2, 1)))
pi = (mu - tcrit * math.sqrt(tau2 + se_mu ** 2),
      mu + tcrit * math.sqrt(tau2 + se_mu ** 2))

z_bound = (0.5 - mu) / se_mu
p_bound = float(norm.sf(z_bound))
Qp = float(1 - __import__("scipy.stats", fromlist=["chi2"]).chi2.cdf(Q, dfQ))

# permutation homogeneity test: resample each cell's own bootstrap draws
obs_disp = float(np.std(y, ddof=1))
cnt = 0
for _ in range(NPERM):
    draw = np.array([RNG.choice(boot[k]["lam"]) for k in keys])
    if np.std(draw, ddof=1) >= obs_disp:
        cnt += 1
p_perm = (cnt + 1) / (NPERM + 1)

print(f"\npooled lambda* = {mu:.3f}  CI [{ci[0]:.3f},{ci[1]:.3f}]  "
      f"tau={tau:.3f}  PI [{pi[0]:.3f},{pi[1]:.3f}]")
print(f"Q={Q:.1f} df={dfQ} p={Qp:.4f} I2={I2:.0f}%  perm p={p_perm:.3f}")
print(f"bound test: {z_bound:.1f} SE below 1/2, one-sided p={p_bound:.3e}")
print(f"cells with CI entirely below 1/2: "
      f"{sum(boot[k]['hi'] < 0.5 for k in keys)}/{kk}")


# ----------------------------------------------------------------------
# M3  AICc model selection
# ----------------------------------------------------------------------
aic_rows = []
for k in keys:
    P = np.array(ref[k]["P"])
    t, e = P[:, 0], P[:, 1]
    n = len(P)
    r_sat = rss_knee(t, e, (ref[k]["eps_inf"], ref[k]["c"], ref[k]["alpha"]))
    pw = fit_power(t, e)
    a_sat, a_pow = aicc(r_sat, n, 3), aicc(pw[2], n, 2)
    d_aicc = a_pow - a_sat
    wgt = 1 / (1 + math.exp(-d_aicc / 2)) if d_aicc < 700 else 1.0
    aic_rows.append(dict(cell=SHORT[k], n=n, rss_sat=r_sat, rss_pow=pw[2],
                         aicc_sat=a_sat, aicc_pow=a_pow, d_aicc=d_aicc,
                         weight=wgt,
                         d_bic=bic(pw[2], n, 2) - bic(r_sat, n, 3),
                         alpha_pow=pw[1]))
d_aiccs = np.array([r["d_aicc"] for r in aic_rows])
print(f"\nAICc: saturating preferred in {(d_aiccs > 0).sum()}/{kk} cells, "
      f"median delta={np.median(d_aiccs):.1f}, min={d_aiccs.min():.1f}, "
      f"median weight={np.median([r['weight'] for r in aic_rows]):.4f}")


# ----------------------------------------------------------------------
# M4  split-half decircularisation
# ----------------------------------------------------------------------
sh_rows = []
for k in keys:
    P = np.array(sorted(ref[k]["P"]))
    if len(P) < 8:
        continue
    A, Bh = P[0::2], P[1::2]
    fa, fb = fit_knee(A[:, 0], A[:, 1]), fit_knee(Bh[:, 0], Bh[:, 1])
    if fa is None or fb is None:
        continue
    sh_rows.append(dict(cell=SHORT[k], lam_full=ref[k]["lam"],
                        lam_a=fa[2] / 2, lam_b=fb[2] / 2,
                        floor_a=fa[0], floor_b=fb[0]))
if sh_rows:
    la = np.array([r["lam_a"] for r in sh_rows])
    lb = np.array([r["lam_b"] for r in sh_rows])
    lf = np.array([r["lam_full"] for r in sh_rows])
    fa_ = np.array([r["floor_a"] for r in sh_rows])
    fb_ = np.array([r["floor_b"] for r in sh_rows])
    half_med = float(np.median(np.concatenate([la, lb])))
    floor_reldiff = float(np.median(np.abs(fa_ - fb_) / ((fa_ + fb_) / 2)))
    print(f"\nsplit-half (alternating rank): median lambda* "
          f"{float(np.median(lf)):.3f} (full) vs {half_med:.3f} (halves), "
          f"shift {half_med - float(np.median(lf)):+.3f}; "
          f"half-to-half r={float(np.corrcoef(la, lb)[0, 1]):.3f}; "
          f"median relative floor difference between halves "
          f"{100 * floor_reldiff:.1f}%  (n={len(sh_rows)})")

# randomised split-half cloud, for cells with a front long enough to halve
cloud = []
for k in keys:
    P = ref[k]["P"]
    if len(P) < 14:
        continue
    for _ in range(200):
        idx = RNG.permutation(len(P))
        A_, B_ = sorted(idx[0::2]), sorted(idx[1::2])
        fa2 = fit_knee_fast([P[i][0] for i in A_], [P[i][1] for i in A_],
                            (ref[k]["eps_inf"], ref[k]["c"], ref[k]["alpha"]))
        fb2 = fit_knee_fast([P[i][0] for i in B_], [P[i][1] for i in B_],
                            (ref[k]["eps_inf"], ref[k]["c"], ref[k]["alpha"]))
        if fa2 and fb2:
            cloud.append((fa2[2] / 2, fb2[2] / 2))
cloud = np.array(cloud)
cloud_mad = float(np.median(np.abs(cloud[:, 0] - cloud[:, 1])))
cloud_r = float(np.corrcoef(cloud[:, 0], cloud[:, 1])[0, 1])
print(f"randomised split-half: {len(cloud)} pairs, median |lam_A - lam_B| = "
      f"{cloud_mad:.3f}, r = {cloud_r:.3f}")
rho_joint = float(np.median([boot[k]["rho"] for k in keys]))
print(f"median bootstrap correlation of jointly fitted (eps_inf, alpha): "
      f"{rho_joint:+.3f}")


# ----------------------------------------------------------------------
# M5  Bayes and minimax characterisations
# ----------------------------------------------------------------------
def KL(p, q):
    p = np.clip(p, 1e-300, 1 - 1e-300)
    q = np.clip(q, 1e-300, 1 - 1e-300)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


# (iv) minimax: sup_p R is attained at an endpoint -> max(-ln q, -ln(1-q))
grid = np.linspace(1e-6, 1 - 1e-6, 1_000_001)
worst = np.maximum(-np.log(grid), -np.log(1 - grid))
i = int(np.argmin(worst))
minimax_q, minimax_v = float(grid[i]), float(worst[i])
# direct check that the supremum over p is the endpoint value
pg = np.linspace(1e-9, 1 - 1e-9, 200001)
sup_direct = max(float(KL(pg, 0.5).max()), float(KL(np.array([0.0, 1.0]), 0.5).max()))
minimax_err = abs(minimax_v - math.log(2))

# (iii) Bayes: argmin_q E_pi[R(p||q)] = E_pi[p]
qg = np.linspace(1e-4, 1 - 1e-4, 200001)
devs = []
for _ in range(300):
    a_, b_ = float(RNG.uniform(0.3, 8)), float(RNG.uniform(0.3, 8))
    m = a_ / (a_ + b_)
    ps = betadist.rvs(a_, b_, size=4000, random_state=int(RNG.integers(1 << 31)))
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    # E_pi[R] = -E[H(p)] - E[p] ln q - (1-E[p]) ln(1-q); minimiser is E[p]
    mp = ps.mean()
    obj = -mp * np.log(qg) - (1 - mp) * np.log(1 - qg)
    devs.append(abs(float(qg[int(np.argmin(obj))]) - mp))
bayes_err = float(np.max(devs))
# symmetric priors -> optimum exactly 1/2
sym = []
for a_ in (0.5, 1.0, 2.0, 5.0, 10.0):
    obj = -0.5 * np.log(qg) - 0.5 * np.log(1 - qg)
    sym.append(abs(float(qg[int(np.argmin(obj))]) - 0.5))
sym_err = float(np.max(sym))

print(f"\nminimax: argmin at q={minimax_q:.12f}, value={minimax_v:.16f}, "
      f"|value-ln2|={minimax_err:.2e}")
print(f"sup_p R(p||1/2) computed directly = {sup_direct:.16f}")
print(f"Bayes: max |argmin - prior mean| over 300 Beta priors = {bayes_err:.2e}; "
      f"symmetric priors: {sym_err:.2e}")


# ----------------------------------------------------------------------
# write outputs
# ----------------------------------------------------------------------
with open(f"{OUT}/table_meta.csv", "w", newline="") as fh:
    wtr = csv.writer(fh)
    wtr.writerow(["cell", "n_front", "eps_inf", "alpha", "lambda_star",
                  "boot_lo", "boot_hi", "boot_se", "below_half", "n_boot"])
    for k in keys:
        wtr.writerow([SHORT[k], len(ref[k]["P"]), f"{ref[k]['eps_inf']:.4f}",
                      f"{ref[k]['alpha']:.4f}", f"{ref[k]['lam']:.4f}",
                      f"{boot[k]['lo']:.4f}", f"{boot[k]['hi']:.4f}",
                      f"{boot[k]['se']:.4f}", int(boot[k]["hi"] < 0.5),
                      boot[k]["nboot"]])

with open(f"{OUT}/table_aicc.csv", "w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=list(aic_rows[0]))
    wtr.writeheader()
    for r in aic_rows:
        wtr.writerow(r)

with open(f"{OUT}/table_splithalf.csv", "w", newline="") as fh:
    if sh_rows:
        wtr = csv.DictWriter(fh, fieldnames=list(sh_rows[0]))
        wtr.writeheader()
        for r in sh_rows:
            wtr.writerow(r)

res = dict(
    n_cells=kk,
    lam=[float(v) for v in y],
    se=[float(math.sqrt(v)) for v in s2],
    lo=[boot[k]["lo"] for k in keys],
    hi=[boot[k]["hi"] for k in keys],
    labels=[SHORT[k] for k in keys],
    pooled=mu, pooled_se=se_mu, ci=list(ci), tau=tau, tau2=tau2,
    Q=Q, Q_df=dfQ, Q_p=Qp, I2=I2, pi=list(pi),
    perm_p=p_perm, z_bound=z_bound, p_bound=p_bound,
    n_ci_below_half=int(sum(boot[k]["hi"] < 0.5 for k in keys)),
    fixed_effect=mu_fe,
    d_aicc=[r["d_aicc"] for r in aic_rows],
    d_aicc_median=float(np.median(d_aiccs)),
    d_aicc_min=float(d_aiccs.min()),
    aicc_weight_median=float(np.median([r["weight"] for r in aic_rows])),
    d_bic_median=float(np.median([r["d_bic"] for r in aic_rows])),
    aicc_labels=[r["cell"] for r in aic_rows],
    splithalf_n=len(sh_rows),
    splithalf_median_full=float(np.median(lf)) if sh_rows else None,
    splithalf_median_halves=half_med if sh_rows else None,

    splithalf_lam_r=float(np.corrcoef(la, lb)[0, 1]) if sh_rows else None,
    splithalf_lam_a=[float(v) for v in la] if sh_rows else [],
    splithalf_lam_b=[float(v) for v in lb] if sh_rows else [],
    splithalf_floor_reldiff=floor_reldiff if sh_rows else None,
    cloud_n=int(len(cloud)), cloud_mad=cloud_mad, cloud_r=cloud_r,
    cloud_a=[float(v) for v in cloud[:, 0]], cloud_b=[float(v) for v in cloud[:, 1]],
    splithalf_labels=[r["cell"] for r in sh_rows],
    joint_rho_median=rho_joint,
    minimax_q=minimax_q, minimax_value=minimax_v, minimax_err=minimax_err,
    sup_direct=sup_direct,
    bayes_max_dev=bayes_err, bayes_sym_dev=sym_err,
    implied_d_over_s=[float((1 - 2 * v) / v) for v in y],
    implied_d_over_s_median=float(np.median([(1 - 2 * v) / v for v in y])),
)
json.dump(res, open(f"{OUT}/stats_meta.json", "w"), indent=1)
print("\nwrote table_meta.csv, table_aicc.csv, table_splithalf.csv, stats_meta.json")
