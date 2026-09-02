#!/usr/bin/env python3
"""
make_fig_ladders.py  --  fig11_ladders.pdf

(a) the price on within-family capacity ladders against the price on
    mixed-population latency fronts: same theory, same estimator, same
    benchmarks, different cell definition;
(b) the ladders in the coordinates of Theorem 1, resolved information against
    log capacity, where a saturating frontier bends and a power law does not;
(c) model selection, saturating against pure power law, for both populations.

Run after llm_ladders.py and stats_llm.py.
"""
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

# stats_ladders.json is a supplied derived input: its producer,
# build/llm_ladders.py, needs the raw leaderboard snapshots that
# licensing prevents us redistributing (see README, Data provenance).
L = json.load(open(f"{DATA}/stats_ladders.json"))
M = json.load(open(f"{OUT}/stats_llm.json"))


def sty(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#d8d8d0", alpha=.9)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(11.2, 3.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=.38)

# ------------------------------------------------------------------- (a) ---
ax = fig.add_subplot(gs[0, 0])
sty(ax)
mixed, lad = M["cells"], L["cells"]
n = 0
for c in mixed:
    ax.plot([c["lo"], c["hi"]], [n, n], color=GREY, lw=1.0, alpha=.8)
    ax.plot([c["lam"]], [n], "o", color=GREY, ms=3.2)
    n += 1
n += 1
sep = n - .5
for c in lad:
    ax.plot([c["lo"], c["hi"]], [n, n], color=BLUE, lw=1.2)
    ax.plot([c["lam"]], [n], "o", color=BLUE, ms=3.6)
    n += 1
ax.axhline(sep, color=BLACK, lw=.6, ls=":")
ax.axvline(.5, color=RED, ls="--", lw=1.0)
ax.axvline(.354, color=ORANGE, lw=1.2)
ax.set_xscale("log")
ax.set_yticks([])
ax.set_xlabel(r"$\lambda^{\star}$")
ax.set_title("(a) same theory, two cell definitions")
p = L["pooled"]
ax.text(.60, .34,
        f"mixed population, {len(mixed)} cells\n"
        rf"  $\tau$={M['flat']['tau']:.2f}, $I^2$={M['flat']['I2']:.0f}%",
        transform=ax.transAxes, va="top", fontsize=6.6, color=GREY)
ax.text(.60, .96,
        f"within family, {len(lad)} ladders\n"
        rf"  pooled {p['mu']:.3f} [{p['ci'][0]:.3f},{p['ci'][1]:.3f}]"
        "\n"
        rf"  $\tau$={p['tau']:.3f}, $I^2$={p['I2']:.0f}%",
        transform=ax.transAxes, va="top", fontsize=6.6, color=BLUE)

# ------------------------------------------------------------------- (b) ---
ax = fig.add_subplot(gs[0, 1])
sty(ax)
COL = [BLUE, GREEN, PURPLE, ORANGE, RED, BLACK]
for i, c in enumerate(lad):
    P = np.array(c["points"])
    x = np.log(P[:, 0])
    S = -np.log(P[:, 1])
    ax.plot(x, S, "o-", color=COL[i % len(COL)], ms=3.4, lw=1.0,
            label=c["label"].replace("eleutherai/", "").replace("facebook/", "")
            .replace("qwen/", "").replace(" / ", " "))
ax.set_xlabel(r"log capacity $x=\ln(\mathrm{params})$")
ax.set_ylabel(r"resolved information $S=-\ln\varepsilon$ (nats)")
ax.set_title("(b) ladders are straight: no knee in reach")
ax.legend(loc="upper left", frameon=False, ncol=1)

# ------------------------------------------------------------------- (c) ---
ax = fig.add_subplot(gs[0, 2])
sty(ax)
dm = [c["d_aicc"] for c in mixed if c.get("d_aicc") is not None]
dl_ = [c["d_aicc"] for c in lad if c.get("d_aicc") is not None]
ax.plot(np.random.default_rng(1).normal(0, .06, len(dm)), dm, "o",
        color=GREY, ms=4, alpha=.85, label="mixed population")
ax.plot(np.random.default_rng(2).normal(1, .06, len(dl_)), dl_, "o",
        color=BLUE, ms=4.4, label="within family")
ax.axhline(0, color=BLACK, lw=.8)
ax.axhspan(-60, 0, color=RED, alpha=.06, lw=0)
ax.set_xticks([0, 1])
ax.set_xticklabels(["mixed", "ladder"])
ax.set_xlim(-.45, 1.45)
ax.set_ylim(min(min(dm), min(dl_)) * 1.08, max(4, max(dm + dl_) * 1.4))
ax.set_ylabel(r"$\Delta\mathrm{AICc}$ (saturating $-$ power law)")
ax.set_title("(c) the power law wins either way")
ax.text(.5, .06, "power law preferred", transform=ax.transAxes,
        ha="center", fontsize=6.6, color=RED)

fig.savefig(f"{OUT}/fig11_ladders.pdf")
print(f"wrote {OUT}/fig11_ladders.pdf")
