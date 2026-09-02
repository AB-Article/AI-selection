#!/usr/bin/env python3
"""
make_fig_retired.py  --  fig12_retired.pdf

(a) evidence for a floor against how far the front has descended, over every
    ladder of the retired suite, coloured by benchmark;
(b) the ladders in the coordinates of Theorem 1: SciQ and LAMBADA bend inside
    the measured range, Winogrande and ARC-Challenge do not;
(c) censoring, language-model ladders and ImageNet cells on one axis: the
    probability of identifying a floor is a function of the range that is left,
    not of the domain;
(d) the price on the saturated ladders against the rest of the study.

Run after analysis_retired.py and retired_stats.py.
"""
import collections
import csv
import json
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 6.8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": .6,
    "grid.linewidth": .4, "lines.linewidth": 1.3, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": .02})

BLACK, RED, BLUE, GREY = "#111111", "#c1121f", "#1b4f9c", "#7a7a70"
ORANGE, GREEN, PURPLE = "#c1531b", "#1a7f5a", "#5b3a8e"
DATA = os.environ.get("DATA", "data")
OUT = os.environ.get("OUT", "results")

S = json.load(open(f"{OUT}/stats_retired.json"))
F = json.load(open(f"{OUT}/stats_retired_final.json"))
cells = {c["label"]: c for c in S["cells"]}

rows = []
for fn in ("table_retired.csv", "table_retired_tokens.csv"):
    for r in csv.DictReader(open(f"{OUT}/{fn}")):
        if int(r["n_front"] or 0) < 6 or not r["eps_min"]:
            continue
        r["eps_min"] = float(r["eps_min"])
        r["d_aicc"] = float(r["d_aicc"]) if r["d_aicc"] else None
        r["S_max"] = -math.log(r["eps_min"])
        rows.append(r)

COL = {"SciQ": RED, "LAMBADA": ORANGE, "ARC-Easy": GREEN, "PIQA": BLUE,
       "Winogrande": PURPLE, "ARC-Challenge": GREY}
ORDER = sorted(F["by_task"], key=lambda t: F["by_task"][t]["eps_min"])


def sty(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#d8d8d0", alpha=.9)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(11.2, 7.4))
gs = fig.add_gridspec(2, 2, wspace=.26, hspace=.34)

# ------------------------------------------------------------------- (a) ---
ax = fig.add_subplot(gs[0, 0])
sty(ax)
for t in ORDER:
    sub = [r for r in rows if r["task"] == t and r["d_aicc"] is not None]
    if not sub:
        continue
    ax.plot([r["S_max"] for r in sub], [r["d_aicc"] for r in sub], "o",
            ms=3.0, mfc=COL.get(t, GREY), mec="none", alpha=.55,
            label=f"{t} ({len(sub)})")
    d = F["by_task"][t]
    ax.plot([d["S_max"]], [d["d_aicc"]], "D", ms=6.5, mfc="none",
            mec=COL.get(t, GREY), mew=1.6)
ax.axhline(0, color=BLACK, lw=.9, ls="--")
ax.set_xlabel(r"resolved information at the best rung, $S_{\max}=-\ln\varepsilon_{\min}$")
ax.set_ylabel(r"$\Delta$AICc  (saturating $-$ power law)")
r = F["rho"]["S_max_daicc"]
ax.set_title(r"(a) evidence for a floor tracks the descent"
             "\n" + rf"$\rho={r[0]:+.2f}$ [{r[1]:+.2f}, {r[2]:+.2f}], "
             rf"cluster bootstrap", loc="left")
ax.legend(frameon=False, loc="upper left", ncol=2, handletextpad=.3)

# ------------------------------------------------------------------- (b) ---
ax = fig.add_subplot(gs[0, 1])
sty(ax)
SUITE = "Pythia v1"
raw = collections.defaultdict(list)
for r in csv.DictReader(open(f"{DATA}/corpus_retired.csv")):
    if r["suite"] != SUITE or int(r["shots"]) or not r["params"]:
        continue
    if int(r["step"] or 0) != 143000:
        continue
    ch, a = float(r["chance"]), float(r["acc"])
    e = 1 - (a - ch) / (1 - ch)
    if 0 < e < 1:
        raw[r["task"]].append((float(r["params"]), e))
for t in ("SciQ", "LAMBADA", "PIQA", "Winogrande"):
    pts, best = [], np.inf
    for x, e in sorted(raw.get(t, [])):
        if e < best - 1e-15:
            pts.append((x, e))
            best = e
    if len(pts) < 6:
        continue
    P = np.array(pts)
    x = np.log(P[:, 0]) - np.log(P[0, 0])
    c = cells.get(f"{SUITE} / {t}")
    ax.plot(x, -np.log(P[:, 1]), marker="o", ms=3.6, color=COL[t],
            label=t + ("" if c else "  (floor at zero)"),
            ls="-" if c else ":")
    if c:
        xs = np.linspace(x.min(), x.max(), 200)
        fit = c["eps_inf"] + (P[0, 1] - c["eps_inf"]) * np.exp(-2 * c["lam"] * xs)
        ax.plot(xs, -np.log(fit), color=COL[t], lw=.9, ls="--", alpha=.75)
ax.set_xlabel(r"log capacity above the smallest rung, $\ln t-\ln t_{0}$")
ax.set_ylabel(r"resolved information $S=-\ln\varepsilon$ (nats)")
ax.set_title(f"(b) one suite ({SUITE}), five orders of capacity\n"
             "markers, data; dashed, fitted softplus", loc="left")
ax.legend(frameon=False, loc="upper left")

# ------------------------------------------------------------------- (c) ---
ax = fig.add_subplot(gs[1, 0])
sty(ax)
cen = collections.defaultdict(list)
for r in csv.DictReader(open(f"{OUT}/table_censoring.csv")):
    cen[r["kind"]].append((float(r["S_max"]), r["identified"] == "True"))
edges = np.array([0, .3, .5, .7, 1.0, 1.5, 4.0])
mid = .5 * (edges[:-1] + edges[1:])
for kind, col, mark in (("LLM ladder", RED, "o"), ("primary cell", BLUE, "s")):
    v = cen.get(kind, [])
    if not v:
        continue
    xs, ys, ns = [], [], []
    for i in range(len(edges) - 1):
        sel = [b for s, b in v if edges[i] <= s < edges[i + 1]]
        if len(sel) < 6:
            continue
        xs.append(mid[i])
        ys.append(np.mean(sel))
        ns.append(len(sel))
    ax.plot(xs, ys, mark + "-", color=col, ms=5,
            label=f"{kind}s ({len(v)} refits)")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"{n}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6, color=col)
ax.set_ylim(-.05, 1.08)
ax.set_xlabel(r"resolved information left after censoring, $S_{\max}$")
ax.set_ylabel("fraction with an interior floor")
g = F["censoring"]["median_gap"]
ax.set_title("(c) censoring the top rungs reproduces the degeneracy\n"
             "on cells whose floor is known to exist", loc="left")
ax.legend(frameon=False, loc="center right")

# ------------------------------------------------------------------- (d) ---
ax = fig.add_subplot(gs[1, 1])
sty(ax)
sat = [c for lab, c in cells.items() if c["task"] in ("SciQ", "LAMBADA")
       and c.get("se")]
sat.sort(key=lambda c: c["lam"])
for i, c in enumerate(sat):
    ax.plot([c["lo"], c["hi"]], [i, i], color=COL[c["task"]], lw=.6, alpha=.35)
    ax.plot([c["lam"]], [i], "o", ms=2.0, color=COL[c["task"]])
p = F["pooled"]["two_stage"]
ax.axvspan(p["ci"][0], p["ci"][1], color=GREEN, alpha=.16)
ax.axvline(p["mu"], color=GREEN, lw=1.5,
           label=rf"saturated ladders, {p['mu']:.3f} "
                 rf"[{p['ci'][0]:.3f},{p['ci'][1]:.3f}]")
ax.axvline(.354, color=ORANGE, lw=1.3, ls="--", label=r"primary corpus, $0.354$")
ax.axvline(.133, color=BLUE, lw=1.1, ls=":",
           label=r"mixed-family ladders, $0.133$")
ax.axvline(.5, color=RED, lw=1.0, ls="--",
           label=r"bound of Proposition 4, $1/2$")
ax.set_xscale("log")
ax.set_yticks([])
ax.set_xlabel(r"canonical price $\lambda^{\star}$ (nats per e-fold)")
ax.set_ylabel(f"{len(sat)} saturated ladders")
ax.set_ylim(-4, len(sat) + 34)
ax.set_title("(d) the price where the benchmark does saturate", loc="left")
ax.legend(frameon=False, loc="upper left", framealpha=0)

fig.savefig(f"{OUT}/fig12_retired.pdf")
print(f"wrote {OUT}/fig12_retired.pdf")
