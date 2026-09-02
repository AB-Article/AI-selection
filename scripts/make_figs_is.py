#!/usr/bin/env python3
"""
make_figs_is.py  --  figures for
"Information yield and its price: a convex duality for the
 accuracy-computation trade-off in AI systems".

Produces
  fig1_mechanism.pdf         softplus / Fermi / susceptibility          (analytic)
  fig4_fronts.pdf            fitted front, collapse, measured Fermi     (corpus)
  fig6_decision.pdf          rule agreement and K-way selection         (results.json)
  fig7_characterisations.pdf minimax, Bayes and quadratic regret        (analytic)
  fig8_stats.pdf             forest plot, AICc, split-half stability    (stats_meta.json)

Run after analysis_main.py and stats_meta.py.
"""
import collections
import csv
import itertools
import json
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 7.0,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": .6,
    "grid.linewidth": .4, "lines.linewidth": 1.3, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": .02})

BLACK, RED, BLUE, GREY, GREEN = "#111111", "#c1121f", "#1b4f9c", "#7a7a70", "#1a7f5a"
FILL = "#dbe6f2"
# resolve data/output locations: works both in the artifacts tree and flat
_D = os.environ.get("DATA", "data")
_OUT = os.environ.get("OUT", "results")
D, OUT = _D, _OUT
os.makedirs(OUT, exist_ok=True)


def sty(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#d8d8d0", alpha=.9)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
# rebuild the identified cells (same code path as analysis_main.py)
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
        r2 = 1 - (r ** 2).sum() / ((ee - ee.mean()) ** 2).sum()
        if best is None or r2 > best[3]:
            best = (p[0], math.exp(p[1]), p[2], r2)
    return best


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
    if len(P) < 6:
        continue
    f = fit_knee([p[0] for p in P], [p[1] for p in P])
    if f is None:
        continue
    ei, c, a, r2 = f
    if not (0 < a < 8) or ei <= 1e-6 or ei >= .999 * min(p[1] for p in P):
        continue
    ref[k] = dict(P=P, eps_inf=ei, c=c, alpha=a, r2=r2, lam=a / 2,
                  t_knee=(c / ei) ** (1 / a))
print(f"cells: {len(ref)}")


# ======================================================================
# FIGURE 1 -- the mechanism
# ======================================================================
fig, ax = plt.subplots(1, 3, figsize=(6.25, 2.05))
al = 0.7
u = np.linspace(-6, 6, 600)
S = -np.log(1 + np.exp(-u))
eta = al / (1 + np.exp(u))
sus = -al ** 2 / (4 * np.cosh(u / 2) ** 2)

ax[0].plot(u / al, S, color=BLACK)
ax[0].axvline(0, color=RED, ls="--", lw=.9)
ax[0].axhline(0, color=GREY, ls=":", lw=.8)
ax[0].annotate(r"ceiling $-\ln\varepsilon_\infty$", (1.4, -.75), color=GREY, fontsize=7.5)
ax[0].annotate("knee", (.35, -3.6), color=RED, fontsize=7.5)
ax[0].set_xlabel(r"$x-\ln t^{\star}$ (e-folds)")
ax[0].set_ylabel(r"$S+\ln\varepsilon_\infty$ (nats)")
ax[0].set_title(r"(a) $S$ is a softplus of $x$")

ax[1].plot(u / al, eta, color=BLACK)
ax[1].axhline(al / 2, color=RED, ls=":", lw=.9)
ax[1].axvline(0, color=RED, ls="--", lw=.9)
ax[1].axhline(al, color=GREY, ls=":", lw=.8)
ax[1].set_ylim(0, al * 1.12)
ax[1].annotate(r"$\alpha$", (-7.4, al * .90), color=GREY, fontsize=8)
ax[1].annotate(r"$\lambda^{\star}=\alpha/2$", (.8, al / 2 + .045), color=RED, fontsize=8)
ax[1].set_xlabel(r"$x-\ln t^{\star}$ (e-folds)")
ax[1].set_ylabel(r"$\eta$ (nats per e-fold)")
ax[1].set_title(r"(b) yield is a Fermi function")

ax[2].plot(u / al, sus, color=BLACK)
ax[2].axvline(0, color=RED, ls="--", lw=.9)
ax[2].set_xlabel(r"$x-\ln t^{\star}$ (e-folds)")
ax[2].set_ylabel(r"$\mathrm{d}\eta/\mathrm{d}x$")
ax[2].set_title(r"(c) $-\alpha^{2}p(1-p)$ peaks at $t^{\star}$")
for a_ in ax:
    sty(a_)
fig.tight_layout(pad=.4)
fig.savefig(f"{OUT}/fig1_mechanism.pdf")
plt.close(fig)


# ======================================================================
# FIGURE 4 -- measured frontiers
# ======================================================================
chords = []
for k, d in ref.items():
    P = d["P"]
    for (t1, e1), (t2, e2) in itertools.combinations(P, 2):
        dx = math.log(t2 / t1)
        if dx < 1.0:
            continue
        eta_c = math.log(e1 / e2) / dx
        v = d["alpha"] * (math.log(math.sqrt(t1 * t2)) - math.log(d["t_knee"]))
        chords.append((v, eta_c / d["alpha"]))
chords = np.array(chords)
print(f"chords >= 1 e-fold: {len(chords)}")

edges = np.linspace(-4, 6, 21)
mid = .5 * (edges[1:] + edges[:-1])
bm, bs, bn = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (chords[:, 0] >= lo) & (chords[:, 0] < hi)
    bm.append(np.median(chords[m, 1]) if m.sum() >= 5 else np.nan)
    bs.append(chords[m, 1].std() / max(math.sqrt(m.sum()), 1) if m.sum() >= 5 else np.nan)
    bn.append(int(m.sum()))
bm, bs = np.array(bm), np.array(bs)
ok = np.isfinite(bm)
pred = 1 / (1 + np.exp(mid))
ss_res = float(((bm[ok] - pred[ok]) ** 2).sum())
ss_tot = float(((bm[ok] - bm[ok].mean()) ** 2).sum())
R2 = 1 - ss_res / ss_tot
r = float(np.corrcoef(bm[ok], pred[ok])[0, 1])
print(f"binned Fermi fit: R2={R2:.3f}, r={r:.3f}, bins={ok.sum()}")

fig, ax = plt.subplots(1, 3, figsize=(6.35, 2.15))
kbig = max(ref, key=lambda z: len(ref[z]["P"]))
d = ref[kbig]
P = np.array(d["P"])
ax[0].loglog(P[:, 0], P[:, 1], "o", ms=3.2, mfc="none", mew=.8, color=BLACK,
             label=f"front ({len(P)} pts)")
tt = np.logspace(math.log10(P[0, 0]) - .4, math.log10(P[-1, 0]) + .4, 300)
ax[0].loglog(tt, d["eps_inf"] + d["c"] * tt ** -d["alpha"], color=RED, lw=1.1,
             label="fitted law")
ax[0].axhline(d["eps_inf"], color=GREY, ls=":", lw=.9)
ax[0].plot([d["t_knee"]], [2 * d["eps_inf"]], "*", ms=11, color=BLUE, zorder=5,
           label=r"$t^{\star}$, $\varepsilon=2\varepsilon_\infty$")
ax[0].set_xlabel("latency (ms)")
ax[0].set_ylabel(r"chance-corrected error $\varepsilon$")
ax[0].set_title("(a) RTX 5090, compiled")
ax[0].legend(loc="upper right", frameon=False)

for k, dd in ref.items():
    Pp = np.array(dd["P"])
    ax[1].plot(Pp[:, 0] / dd["t_knee"], Pp[:, 1] / dd["eps_inf"], "-", lw=.8,
               alpha=.8, color=BLACK if k[0] == "timm" else BLUE)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].axhline(2, color=RED, ls="--", lw=.9)
ax[1].axvline(1, color=RED, ls="--", lw=.9)
ax[1].set_xlabel(r"$t/t^{\star}$")
ax[1].set_ylabel(r"$\varepsilon/\varepsilon_\infty$")
ax[1].set_title("(b) twelve cells, collapsed")

ax[2].plot(chords[:, 0], chords[:, 1], ".", ms=1.0, alpha=.10, color=GREY)
ax[2].errorbar(mid[ok], bm[ok], yerr=bs[ok], fmt="o", ms=3.2, lw=.8,
               color=BLACK, capsize=1.6, label="measured (binned)")
vv = np.linspace(-4, 6, 300)
ax[2].plot(vv, 1 / (1 + np.exp(vv)), color=RED, lw=1.2, label="Fermi prediction")
ax[2].set_ylim(-.05, 1.18)
ax[2].set_xlabel(r"$v=\alpha\ln(t/t^{\star})$")
ax[2].set_ylabel(r"$\eta/\alpha$")
ax[2].set_title(rf"(c) $R^{{2}}={R2:.3f}$, {len(chords)} chords")
ax[2].legend(loc="upper right", frameon=False)
for a_ in ax:
    sty(a_)
fig.tight_layout(pad=.4)
fig.savefig(f"{OUT}/fig4_fronts.pdf")
plt.close(fig)


# ======================================================================
# FIGURE 6 -- decision performance
# ======================================================================
res = json.load(open(f"{OUT}/results.json"))
names = ["Leave-one-cell-out knee price", "Universal knee price $\\lambda=0.35$",
         "Fixed $\\lambda=0.3$", "Fixed $\\lambda=0.4$",
         "Leave-one-task-out knee price", "Fixed $\\lambda=0.5$",
         "Fixed $\\lambda=0.25$", "Accuracy per unit time ($\\varepsilon t$)",
         "Always the faster system", "Fixed $\\lambda=0.2$",
         "Fixed $\\lambda=0.166$", "Coin flip",
         "Always the more accurate system"]
lbl = ["leave-one-cell-out", r"universal $\hat\lambda{=}0.35$", r"$\lambda{=}0.30$",
       r"$\lambda{=}0.40$", "leave-one-task-out", r"$\lambda{=}0.50$",
       r"$\lambda{=}0.25$", r"$\lambda{=}1$ (info per unit time)", "always faster",
       r"$\lambda{=}0.20$", r"MAI $\lambda{=}0.166$", "coin flip", "always accurate"]
v = [res["rules"][n]["acc"] for n in names]
o = np.argsort(v)
cols = [RED if names[i].startswith(("Leave-one-cell", "Universal")) else "#b9b9b0"
        for i in o]
fig, ax = plt.subplots(1, 2, figsize=(6.35, 2.5))
ax[0].barh(range(len(v)), [v[i] for i in o], color=cols, height=.72,
           edgecolor=BLACK, linewidth=.4)
ax[0].set_yticks(range(len(v)))
ax[0].set_yticklabels([lbl[i] for i in o], fontsize=6.8)
ax[0].set_xlim(20, 100)
ax[0].set_xlabel("agreement with fitted theory (%)")
ax[0].set_title("(a) two-system decision, 5459 pairs")
K = sorted(int(x) for x in res["nway"])
ax[1].plot(K, [res["nway"][str(k)]["acc"] for k in K], "o-", color=RED, ms=4, lw=1.2)
ax[1].set_xscale("log")
ax[1].set_xticks(K)
ax[1].set_xticklabels(K)
ax[1].set_ylim(75, 100)
ax[1].set_xlabel("number of candidate systems $K$")
ax[1].set_ylabel("same system as fitted theory (%)")
ax[1].set_title("(b) $K$-way selection")
for a_ in ax:
    sty(a_)
fig.tight_layout(pad=.4)
fig.savefig(f"{OUT}/fig6_decision.pdf")
plt.close(fig)


# ======================================================================
# FIGURE 7 -- the three decision-theoretic characterisations
# ======================================================================
def KL(p, q):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


fig, ax = plt.subplots(1, 3, figsize=(6.35, 2.05))
q = np.linspace(.02, .98, 1200)
for d, col, ls, lw in ((0.499, BLACK, "-", 1.3), (0.30, BLUE, "--", 1.1),
                       (0.15, GREEN, "-.", 1.1)):
    plo, phi = .5 - d, .5 + d
    worst = np.maximum(KL(plo, q), KL(phi, q))
    ax[0].plot(q, worst, color=col, ls=ls, lw=lw,
               label=rf"$\delta={d:.2f}$")
    ax[0].plot([.5], [math.log(2) - float(-(plo * np.log(plo) +
               (1 - plo) * np.log(1 - plo)))], "o", ms=3.6, color=col)
ax[0].axhline(math.log(2), color=GREY, ls=":", lw=.9)
ax[0].axvline(.5, color=RED, ls="--", lw=.9)
ax[0].annotate(r"$\ln 2$", (.80, math.log(2) + .10), color=GREY, fontsize=7.5)
ax[0].set_ylim(0, 2.6)
ax[0].set_xlabel(r"declared $\hat p=\hat\lambda/\alpha$")
ax[0].set_ylabel("worst-case regret (nats)")
ax[0].set_title("(a) minimax price")
ax[0].legend(frameon=False, loc="upper center", fontsize=6.2,
             title=r"$\mathcal{P}=[0.5-\delta,\ 0.5+\delta]$",
             title_fontsize=6.2)

pg = np.linspace(1e-4, 1 - 1e-4, 4000)
for a_, b_, col, ls in ((2, 2, BLACK, "-"), (5, 5, BLUE, "--"), (0.7, 0.7, GREEN, "-.")):
    w = pg ** (a_ - 1) * (1 - pg) ** (b_ - 1)
    w /= w.sum()
    m = float((w * pg).sum())
    bayes = np.array([float((w * KL(pg, qq)).sum()) for qq in q])
    ax[1].plot(q, bayes, color=col, ls=ls, lw=1.1,
               label=rf"$\mathrm{{Beta}}({a_:g},{b_:g})$")
ax[1].axvline(.5, color=RED, ls="--", lw=.9)
ax[1].set_ylim(0, 1.4)
ax[1].set_xlabel(r"declared $\hat p$")
ax[1].set_ylabel("Bayes regret (nats)")
ax[1].set_title("(b) Bayes price under symmetry")
ax[1].legend(frameon=False, loc="upper center")

dp = np.linspace(0, .3, 400)
ax[2].plot(dp, KL(.5 + dp, .5), color=BLACK, label="exact KL")
ax[2].plot(dp, 2 * dp ** 2, color=RED, ls="--", label=r"$2(\Delta p)^{2}$")
ax[2].axvspan(0, .075, color=FILL, alpha=.9, lw=0)
ax[2].set_xlabel(r"$|p-\hat p|$")
ax[2].set_ylabel("regret (nats)")
ax[2].set_title("(c) quadratic near $\\lambda^{\\star}$")
ax[2].legend(frameon=False, loc="upper left")
for a_ in ax:
    sty(a_)
fig.tight_layout(pad=.4)
fig.savefig(f"{OUT}/fig7_characterisations.pdf")
plt.close(fig)


# ======================================================================
# FIGURE 8 -- meta-analysis, model selection, split-half stability
# ======================================================================
M = json.load(open(f"{OUT}/stats_meta.json"))
lab = M["labels"]
y = np.array(M["lam"])
lo = np.array(M["lo"])
hi = np.array(M["hi"])
n = len(y)

fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.7),
                       gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
yy = np.arange(n)[::-1]
ax[0].axvspan(M["ci"][0], M["ci"][1], color="#f3d9c9", alpha=.9, lw=0)
ax[0].axvline(M["pooled"], color="#c1531b", lw=1.2)
ax[0].axvline(.5, color=RED, ls="--", lw=1.0)
ax[0].hlines(yy, lo, hi, color=BLACK, lw=.9)
ax[0].plot(y, yy, "s", ms=3.4, color=BLACK)
ax[0].plot(M["pi"], [-1.1, -1.1], color="#c1531b", lw=2.0,
           solid_capstyle="butt")
ax[0].plot([M["pooled"]], [-1.1], "D", ms=4, color="#c1531b")
ax[0].set_yticks(list(yy) + [-1.1])
ax[0].set_yticklabels([s.replace("amp-fp16", "fp16") for s in lab] +
                      ["pooled (95% PI)"], fontsize=6.2)
ax[0].set_ylim(-2.0, n - .3)
ax[0].set_xlim(0.10, 0.72)
ax[0].set_xlabel(r"$\lambda^{\star}=\alpha/2$")
ax[0].set_title("(a) random-effects pooling")
ax[0].annotate(r"$\lambda^{\star}=1/2$", (.505, n - 1.6), color=RED, fontsize=7,
               rotation=90, va="top")

da = np.array(M["d_aicc"])
order = np.argsort(da)
ax[1].barh(np.arange(len(da)), da[order], color="#b9b9b0", edgecolor=BLACK,
           linewidth=.4, height=.72)
ax[1].axvline(0, color=BLACK, lw=.8)
ax[1].axvline(10, color=RED, ls="--", lw=.9)
ax[1].set_yticks([])
ax[1].set_xlabel(r"$\Delta$AICc vs. power law")
ax[1].set_title("(b) model selection")
ax[1].annotate(r"$\Delta=10$", (11, .2), color=RED, fontsize=7)

ca, cb = np.array(M["cloud_a"]), np.array(M["cloud_b"])
ax[2].plot(ca, cb, ".", ms=1.6, alpha=.16, color=BLACK)
lim = (.10, .95)
ax[2].plot(lim, lim, color=RED, lw=.9)
ax[2].set_xlim(*lim)
ax[2].set_ylim(*lim)
ax[2].set_xlabel(r"$\lambda^{\star}$, half A")
ax[2].set_ylabel(r"$\lambda^{\star}$, half B")
ax[2].set_title(rf"(c) split-half stability")
for a_ in ax:
    sty(a_)
fig.tight_layout(pad=.4)
fig.savefig(f"{OUT}/fig8_stats.pdf")
plt.close(fig)

json.dump(dict(chords=int(len(chords)), fermi_R2=R2, fermi_r=r,
               fermi_bins=int(ok.sum())),
          open(f"{OUT}/fig_stats.json", "w"), indent=1)
print("figures written to", OUT)
