#!/usr/bin/env python3
r"""Final numbers and Figure 9 for the revised manuscript.

Everything is derived from Table 7 of the manuscript (eps_inf, alpha, t*)
plus externally published prices.  Nothing is invented.
"""
import json
import os
import math

DATA = os.environ.get("DATA", "data")
OUT = os.environ.get("OUT", "results")
os.makedirs(OUT, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RNG = np.random.default_rng(20260824)

# Table 7, verbatim.  (name, task, eps_inf, alpha, tstar_ms, device)
CELLS = [
    ("RTX Pro 6000 / compile bf16", "Classification", 0.0935, 0.538, 0.040, "RTX PRO 6000"),
    ("RTX Pro 6000 / eager fp16",   "Classification", 0.0981, 0.648, 0.052, "RTX PRO 6000"),
    ("CPU i7-12700H / compile fp32", "Classification", 0.0994, 0.655, 10.76, None),
    ("RTX 3090 / eager fp16",       "Classification", 0.0994, 0.657, 0.112, "RTX 3090"),
    ("RTX 4090 / eager fp16",       "Classification", 0.0994, 0.688, 0.063, "RTX 4090"),
    ("RTX 5090 / eager fp16",       "Classification", 0.0994, 0.688, 0.044, "RTX 5090"),
    ("RTX 4090 / compile fp16",     "Classification", 0.0994, 0.688, 0.041, "RTX 4090"),
    ("RTX 5090 / compile fp16",     "Classification", 0.0994, 0.717, 0.025, "RTX 5090"),
    ("CPU ONNX (oriented det.)",    "Oriented det.",  0.1807, 0.719, 12.36, None),
    ("CPU ONNX (pose est.)",        "Pose est.",      0.2789, 0.889, 40.14, None),
    ("CPU i9-10940X / compile fp32", "Classification", 0.0994, 0.951, 7.195, None),
    ("CPU ONNX (object det.)",      "Object det.",    0.4316, 1.052, 23.28, None),
]
PRICES = {"RTX 4090": 0.43, "RTX 5090": 0.60, "RTX 3090": 0.15, "RTX PRO 6000": 2.20}
V_LO, V_MID, V_HI = 0.01, 0.05, 0.15          # USD per human image label


def eps_of_t(t, a, c, e):
    return e + c * np.power(t, -a)


def eta_of_t(t, a, c, e):
    r = c * np.power(t, -a)
    return a * r / (e + r)


def t_econ(rho, a, c):
    return (a * c / rho) ** (1.0 / (a + 1.0))


def lam_econ(rho, a, c, e):
    return eta_of_t(t_econ(rho, a, c), a, c, e)


N = {}
rows, mult_all, mult_gpu, vstars, rstars, lam_a1 = [], [], [], [], [], []
for name, task, e, a, ts_ms, dev in CELLS:
    ts = ts_ms * 1e-3
    c = e * ts ** a
    ae = a * e
    mult = 1.0 / ae
    mult_all.append(mult)
    if dev is None:
        rows.append([name, task, None, ts_ms, ae, mult, None, None])
        continue
    w = PRICES[dev] / 3600.0
    vs = w * ts / ae
    rs = ae / ts
    la = lam_econ(w / V_MID, a, c, e)
    mult_gpu.append(mult); vstars.append(vs); rstars.append(rs); lam_a1.append(la)
    rows.append([name, task, PRICES[dev], ts_ms, ae, mult, vs, la])

N["mult_median_all"] = float(np.median(mult_all))
N["mult_min_all"], N["mult_max_all"] = float(min(mult_all)), float(max(mult_all))
N["mult_median_cell"] = 1.0 / (0.688 * 0.0994)
N["vstar_median"] = float(np.median(vstars))
N["vstar_min"], N["vstar_max"] = float(min(vstars)), float(max(vstars))
N["ratio_median"] = float(np.median(vstars)) / V_MID
N["lam_a1_min"], N["lam_a1_max"] = float(min(lam_a1)), float(max(lam_a1))
N["lam_a1_median"] = float(np.median(lam_a1))
N["rhostar_median"] = float(np.median(rstars))
N["rho_a1_median"] = float(np.median([PRICES[d] / 3600.0 / V_MID
                                      for *_, d in CELLS if d]))
N["decades"] = math.log10(N["rhostar_median"] / N["rho_a1_median"])
N["n_gpu"] = len(vstars)
# knee that WOULD make alpha/2 optimal at a 5-cent label, median GPU cell
N["tstar_needed_s"] = V_MID * (0.688 * 0.0994) / (0.43 / 3600.0)
N["tstar_actual_s"] = 0.063e-3
N["tstar_factor"] = N["tstar_needed_s"] / N["tstar_actual_s"]

# --- capacity reading: alpha/2 is the shadow price iff the budget is the knee
N["capacity_qps_per_device_at_knee"] = 1.0 / 0.063e-3

# --- LaTeX table body -------------------------------------------------
def sci(x):
    m, ex = f"{x:.2e}".split("e")
    return f"${m}\\times10^{{{int(ex)}}}$"


body = []
for name, task, w, ts_ms, ae, mult, vs, la in rows:
    if w is None:
        body.append(f"{name} & {task} & --- & {ts_ms:.3f} & {ae:.4f} & "
                    f"{mult:.1f} & --- & --- \\\\")
    else:
        body.append(f"{name} & {task} & {w:.2f} & {ts_ms:.3f} & {ae:.4f} & "
                    f"{mult:.1f} & {sci(vs)} & {la:.4f} \\\\")
N["table_body"] = body

# --- tangency: agreement of the two verdicts vs pair separation, at rho* ---
a, e = 0.688, 0.0994
ts = 0.063e-3
c = e * ts ** a
rs = a * e / ts
tang = {}
for sep in (1.05, 1.5, 2.0, 4.0, 10.0, 100.0):
    tb = ts / math.sqrt(sep) * np.exp(RNG.normal(0, 0.35, 40000))
    ta = tb * sep
    ea, eb = eps_of_t(ta, a, c, e), eps_of_t(tb, a, c, e)
    econ = rs < (eb - ea) / (ta - tb)
    price = (np.log(eb / ea) / np.log(ta / tb)) > a / 2
    tang[sep] = float(np.mean(econ == price))
N["tangency"] = tang

# ======================================================================
# Figure 9
# ======================================================================
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7,
                     "font.family": "serif", "mathtext.fontset": "dejavuserif"})
fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.3))

# (a) lambda_ec(rho) for every priced cell
axa = ax[0]
rho = np.logspace(-4, 7, 500)
for name, task, e_, a_, ts_ms, dev in CELLS:
    if dev is None:
        continue
    ts_ = ts_ms * 1e-3
    c_ = e_ * ts_ ** a_
    axa.plot(rho, lam_econ(rho, a_, c_, e_), lw=1.0, alpha=0.75)
axa.axhspan(0.263, 0.446, color="orange", alpha=0.18, lw=0)
axa.axhline(0.344, color="darkorange", lw=1.2)
axa.axvline(N["rho_a1_median"], color="crimson", ls="--", lw=1.2)
axa.set_xscale("log")
axa.set_xlabel(r"cost-to-value ratio $\varrho=w/v$  (s$^{-1}$)")
axa.set_ylabel(r"economically optimal price $\lambda_{\mathrm{ec}}(\varrho)$")
axa.set_title("(a) The realised price is a bijection in $\\varrho$", fontsize=8.5)
axa.text(N["rho_a1_median"] * 1.6, 0.95, "\\$0.05 label\nanchor", color="crimson",
         fontsize=7, va="top")
axa.text(2e5, 0.36, r"$\lambda^\star$ prediction interval", color="darkorange",
         fontsize=7, ha="right")
axa.annotate("", xy=(N["rhostar_median"], 0.344), xytext=(N["rho_a1_median"], 0.344),
             arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9))
axa.text(math.sqrt(N["rhostar_median"] * N["rho_a1_median"]), 0.30,
         f"{N['decades']:.1f} decades", ha="center", fontsize=7, color="0.25")
axa.set_ylim(0, 1.12)
axa.set_xlim(1e-4, 1e7)

# (b) implied v* against published label cost
axb = ax[1]
labels = [r[0].replace(" / ", "\n") for r in rows if r[2] is not None]
vals = [r[6] for r in rows if r[2] is not None]
y = np.arange(len(vals))
axb.barh(y, vals, color="steelblue", height=0.6)
axb.axvspan(V_LO, V_HI, color="crimson", alpha=0.18, lw=0)
axb.axvline(V_MID, color="crimson", lw=1.2)
axb.set_yticks(y)
axb.set_yticklabels(labels, fontsize=5.6)
axb.set_xscale("log")
axb.set_xlim(1e-8, 1e0)
axb.set_xlabel(r"value of one correct answer (USD)")
axb.set_title(r"(b) $v^\star$ implied by $\lambda^\star$ vs. human labelling",
              fontsize=8.5)
axb.text(V_MID * 1.5, len(vals) - 0.4, "published\nlabel cost", color="crimson",
         fontsize=7, va="top")
axb.text(2e-7, 0.6, f"gap $\\approx${N['ratio_median']:.0e}", fontsize=7, color="0.25")

# (c) agreement of the two verdicts at rho*, vs pair separation
axc = ax[2]
seps = list(tang)
axc.plot(seps, [tang[s] for s in seps], "o-", color="seagreen", lw=1.2, ms=4)
axc.axvline(1.5, color="0.4", ls=":", lw=1.0)
axc.set_xscale("log")
axc.set_xlabel(r"pair separation $t_A/t_B$")
axc.set_ylabel("agreement of the two verdicts")
axc.set_ylim(0.7, 1.02)
axc.set_title(r"(c) The criteria are tangent at $t^\star$", fontsize=8.5)
axc.text(1.55, 0.75, "paper's $\\ln 1.5$\nthreshold", fontsize=7, color="0.35")

for a_ in ax:
    a_.grid(alpha=0.25, lw=0.5)
    a_.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_econ.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_econ.png", dpi=170, bbox_inches="tight")

with open(f"{OUT}/final_numbers.json", "w") as fh:
    json.dump(N, fh, indent=2, default=float)
print(json.dumps({k: v for k, v in N.items() if k != "table_body"},
                 indent=2, default=float))
print("\n".join(N["table_body"]))
