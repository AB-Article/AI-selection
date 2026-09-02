#!/usr/bin/env python3
"""
make_figs_ext.py  --  fig9_extended.pdf

(a) forest plot of the canonical price across all identified cells, coloured by
    evidence group, with the random-effects pooled estimate and the bound of
    Proposition 13;
(b) pooled price by task;
(c) why the LLM cells behave differently: the information span of each frontier,
    max(S) - min(S) in nats, which is what a three-parameter saturating law
    needs in order to be identified at all.

Run after stats_extended.py.
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 6.6,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": .6,
    "grid.linewidth": .4, "lines.linewidth": 1.3, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": .02})

BLACK, RED, BLUE, GREY, GREEN = "#111111", "#c1121f", "#1b4f9c", "#7a7a70", "#1a7f5a"
ORANGE = "#c1531b"
_OUT = os.environ.get("OUT", "results")
OUT = _OUT

M = json.load(open(f"{OUT}/stats_extended.json"))
cells = M["cells"]

COL = {"Primary vision inference": BLACK,
       "Extended cross-domain": BLUE,
       "LLM inference": RED}


def sty(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#d8d8d0", alpha=.9)
    ax.set_axisbelow(True)


fig, ax = plt.subplots(1, 3, figsize=(7.1, 3.6),
                       gridspec_kw=dict(width_ratios=[1.55, 1.0, 1.0]))

# ---------------------------------------------------------------- (a) ------
order = sorted(range(len(cells)), key=lambda i: cells[i]["lam"])
y = np.arange(len(order))[::-1]
for yy, i in zip(y, order):
    c = cells[i]
    col = COL.get(c["group"], GREY)
    ax[0].hlines(yy, c["lo"], min(c["hi"], 2.6), color=col, lw=.9)
    ax[0].plot(c["lam"], yy, "s", ms=3.2, color=col)
    if c["hi"] > 2.6:
        ax[0].plot(2.6, yy, ">", ms=3, color=col)
A = M["all"]
ax[0].axvspan(A["ci"][0], A["ci"][1], color="#f3d9c9", alpha=.85, lw=0, zorder=0)
ax[0].axvline(A["mu"], color=ORANGE, lw=1.2, zorder=1)
ax[0].axvline(0.5, color=RED, ls="--", lw=1.0, zorder=1)
ax[0].plot(A["pi"], [-1.3, -1.3], color=ORANGE, lw=2.2, solid_capstyle="butt")
ax[0].plot([A["mu"]], [-1.3], "D", ms=4, color=ORANGE)
ax[0].set_yticks(list(y) + [-1.3])
ax[0].set_yticklabels([cells[i]["label"] for i in order] +
                      ["pooled (95\\% PI)"], fontsize=5.6)
ax[0].set_xscale("log")
ax[0].set_xlim(0.02, 2.8)
ax[0].set_xticks([0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0])
ax[0].set_xticklabels(["0.05", "0.1", "0.2", "0.35", "0.5", "1", "2"])
ax[0].set_ylim(-2.4, len(order) + 4.2)
ax[0].set_xlabel(r"$\lambda^{\star}=\alpha/2$ (log scale)")
ax[0].set_title("(a) every identified cell")
h = [plt.Line2D([], [], color=COL[g], marker="s", ms=3.4, lw=.9, label=g)
     for g in ("Primary vision inference", "Extended cross-domain",
               "LLM inference")]
ax[0].legend(handles=h, loc="upper center", frameon=False,
              bbox_to_anchor=(0.5, 1.005), ncol=1, handletextpad=.5)

# ---------------------------------------------------------------- (b) ------
bt = M["by_task"]
names = sorted(bt, key=lambda t: bt[t]["mu"])
yy = np.arange(len(names))[::-1]
for j, t in zip(yy, names):
    p = bt[t]
    ax[1].hlines(j, p["ci"][0], p["ci"][1], color=BLACK, lw=1.0)
    ax[1].plot(p["mu"], j, "o", ms=4.2, color=BLACK)
    ax[1].annotate(f"$k={p['k']}$", (p["ci"][1] + .02, j), fontsize=6.4,
                   va="center", color=GREY)
ax[1].axvline(A["mu"], color=ORANGE, lw=1.2)
ax[1].axvline(0.5, color=RED, ls="--", lw=1.0)
ax[1].set_yticks(yy)
ax[1].set_yticklabels(names, fontsize=6.4)
ax[1].set_xlim(0, 0.95)
ax[1].set_ylim(-.8, len(names) - .2)
ax[1].set_xlabel(r"pooled $\lambda^{\star}$")
ax[1].set_title("(b) by task")

# ---------------------------------------------------------------- (c) ------
import csv
rows = list(csv.DictReader(open(f"{OUT}/table_extended.csv")))
for r in rows:
    if not r.get("S_range") or not r.get("R2"):
        continue
    llm = r["task"] == "LLM inference"
    ident = r["status"] == "identified"
    ax[2].plot(float(r["S_range"]), float(r["R2"]),
               "o" if ident else "x",
               ms=4.4 if ident else 4.8,
               mfc=(RED if llm else "#b9b9b0") if ident else "none",
               mec=RED if llm else BLACK, mew=.9)
ax[2].axvline(0.5, color=GREY, ls=":", lw=.9)
ax[2].set_xlim(0, 2.5)
ax[2].set_ylim(0.35, 1.03)
ax[2].set_xlabel(r"front span $\Delta S$ (nats)")
ax[2].set_ylabel(r"$R^{2}$ of the saturating fit")
ax[2].set_title("(c) why the LLM cells fail")
h2 = [plt.Line2D([], [], ls="", marker="o", ms=4.4, mfc="#b9b9b0", mec=BLACK,
                 label="vision, speech"),
      plt.Line2D([], [], ls="", marker="o", ms=4.4, mfc=RED, mec=RED,
                 label="LLM"),
      plt.Line2D([], [], ls="", marker="x", ms=4.8, mec=BLACK,
                 label="not identified")]
ax[2].legend(handles=h2, loc="center left", frameon=False)

for a_ in ax:
    sty(a_)
fig.tight_layout(pad=.5)
fig.savefig(f"{OUT}/fig9_extended.pdf")
plt.close(fig)
print("wrote fig9_extended.pdf")
