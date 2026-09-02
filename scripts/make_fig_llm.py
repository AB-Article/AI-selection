#!/usr/bin/env python3
"""
make_fig_llm.py  --  fig10_llm.pdf

(a) forest plot of lambda* for every identified task-resolved LLM cell, with
    the primary pooled price and the bound of Proposition 13 for reference;
(b) information span of the front against fit quality, for every candidate
    cell with a front of at least six points, with the four aggregate-metric
    cells of the previous version marked -- the span problem is fixed, the
    identification problem is not;
(c) lambda* against front length, which is where the remaining scatter lives.

Run after stats_llm.py.
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

BLACK, RED, BLUE, GREY = "#111111", "#c1121f", "#1b4f9c", "#7a7a70"
ORANGE, GREEN = "#c1531b", "#1a7f5a"
OUT = os.environ.get("OUT", "results")

M = json.load(open(f"{OUT}/stats_llm.json"))
cells = M["cells"]
try:
    E = json.load(open(f"{OUT}/stats_extended.json"))
except FileNotFoundError:
    E = None

SRC_COL = {"llm-perf-task": BLACK, "llm-inference-bench": BLUE,
           "llama.cpp": RED, "bench360": GREEN}
SRC_LAB = {"llm-perf-task": "LLM-Perf, task-resolved",
           "llm-inference-bench": "LLM-Inference-Bench",
           "llama.cpp": "llama.cpp quantisation",
           "bench360": "Bench360"}


def sty(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#d8d8d0", alpha=.9)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(11.2, 3.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=.42)

# ------------------------------------------------------------------- (a) ---
ax = fig.add_subplot(gs[0, 0])
sty(ax)
y = np.arange(len(cells))
for i, c in enumerate(cells):
    col = SRC_COL.get(c["source"], GREY)
    ax.plot([c["lo"], c["hi"]], [i, i], color=col, lw=1.1, alpha=.85)
    ax.plot([c["lam"]], [i], "o", color=col, ms=3.4, zorder=3)
ax.axvline(.5, color=RED, ls="--", lw=1.0)
ax.axvline(.354, color=ORANGE, lw=1.2)
ax.axvspan(.325, .384, color=ORANGE, alpha=.16, lw=0)
ax.set_yticks(y)
ax.set_yticklabels([c["label"].replace("llm-perf | ", "")
                    .replace("llm-bench | ", "").replace(" GPU", "")
                    .replace("llama.cpp | 1x RTX 4090 | ", "llama.cpp / ")
                    for c in cells], fontsize=5.6)
ax.set_xscale("log")
ax.set_xlabel(r"canonical price $\lambda^{\star}$")
ax.set_title(f"(a) {len(cells)} identified task-resolved LLM cells")
h = [plt.Line2D([], [], color=SRC_COL[k], lw=1.4, marker="o", ms=3.2,
                label=SRC_LAB[k]) for k in SRC_LAB if
     any(c["source"] == k for c in cells)]
h += [plt.Line2D([], [], color=ORANGE, lw=1.4, label="primary pooled 0.354"),
      plt.Line2D([], [], color=RED, ls="--", lw=1.0, label=r"bound $1/2$")]
ax.legend(handles=h, loc="lower right", frameon=False)

# ------------------------------------------------------------------- (b) ---
ax = fig.add_subplot(gs[0, 1])
sty(ax)
rows = []
import csv
for r in csv.DictReader(open(f"{OUT}/table_llm.csv")):
    if r["n_front"] and int(r["n_front"]) >= 6 and r["S_range"] and r["R2"]:
        rows.append((float(r["S_range"]), float(r["R2"]),
                     r["status"] == "identified", r["source"]))
for S, r2, ok, src in rows:
    ax.plot([S], [r2], "o" if ok else "x",
            color=SRC_COL.get(src, GREY), ms=4 if ok else 4.4,
            mfc=SRC_COL.get(src, GREY) if ok else "none", alpha=.85)
if E:
    for c in E["llm"].values():
        if c.get("S_range") and c.get("R2"):
            ax.plot([c["S_range"]], [c["R2"]], "s", color=GREY, ms=5.2,
                    mfc="none")
    ax.plot([], [], "s", color=GREY, mfc="none", ms=5.2,
            label="aggregate metric (previous)")
ax.axvline(0.609, color=GREY, ls=":", lw=1.0)
ax.text(0.62, 0.33, "vision/speech\nmedian span", fontsize=6, color=GREY)
ax.plot([], [], "o", color=BLACK, ms=4, label="identified")
ax.plot([], [], "x", color=BLACK, ms=4.4, label="not identified")
ax.set_xlabel("information span of the front (nats)")
ax.set_ylabel(r"$R^{2}$ of the saturating fit")
ax.set_title("(b) span is no longer binding")
ax.legend(loc="lower right", frameon=False)

# ------------------------------------------------------------------- (c) ---
ax = fig.add_subplot(gs[0, 2])
sty(ax)
for c in cells:
    ax.plot([c["n_front"]], [c["lam"]], "o", color=SRC_COL.get(c["source"], GREY),
            ms=4.2, alpha=.9)
ax.axhline(.5, color=RED, ls="--", lw=1.0)
ax.axhline(.354, color=ORANGE, lw=1.2)
ax.set_yscale("log")
ax.set_xlabel("points on the Pareto front")
ax.set_ylabel(r"$\lambda^{\star}$")
ax.set_title("(c) scatter against front length")

fig.savefig(f"{OUT}/fig10_llm.pdf")
print(f"wrote {OUT}/fig10_llm.pdf")
