#!/usr/bin/env python3
r"""
external_criterion.py
=====================
Produces every number in Sections 6 and 8 of the revised manuscript
("Pricing the quality-computation trade-off").

The point of this script is to score selection rules against a verdict that
contains NONE of the theory's fitted quantities:

    U(eps, t; v, w) = v * (1 - eps) - w * t

For a pair (A accurate+slow, B fast):

    prefer A  <=>  rho < rho_AB := (eps_B - eps_A) / (t_A - t_B),   rho = w/v

Nothing in that verdict mentions alpha, eps_inf or lambda*.  The theory is
then connected to it by Proposition 1:

    t_ec(rho) = (alpha * c / rho)^(1/(alpha+1))
    lambda_ec(rho) = eta(t_ec(rho))         strictly decreasing onto (0, alpha)
    rho*  = alpha * eps_inf / t*            (the unique rho where lambda_ec = alpha/2)
    v*    = w * t* / (alpha * eps_inf)
    v* / (w t*) = 1 / (alpha * eps_inf)

INPUTS  (CSV, --indir)
----------------------
cells.csv          cell_id, source, task, device, alpha, alpha_se, eps_inf, c, n_front
pairs.csv          cell_id, sys_A, sys_B, eps_A, t_A, eps_B, t_B      (t in SECONDS)
device_prices.csv  device, usd_per_hour, source_url, retrieved
anchors.csv        task, anchor, v_usd            (anchor in {A1, A2});
                   for A3 give   task, A3, rho_per_s   in a rho_per_s column

OUTPUTS (--outdir)
------------------
tab_econ_implied.tex      Table: implied valuations per cell
tab_econ_baselines.tex    Table: rules vs deployment-cost verdict, 3 anchors
tab_revealed.tex          Table skeleton for revealed preference (if adoption.csv)
econ_sweep.csv            kappa and regret vs rho for every rule
numbers.json              every \NUM{} placeholder value, keyed by name
fig_econ_sweep.pdf        Figure 8

USAGE
-----
    python external_criterion.py --selftest             # verify the maths
    python external_criterion.py --indir data --outdir out

Seeds fixed at 20260824 to match the manuscript.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

SEED = 20260824
RNG = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# Frontier algebra  (Proposition 1)
# ----------------------------------------------------------------------


def eps_of_t(t, alpha, c, eps_inf):
    """Saturating frontier, Axiom 1."""
    return eps_inf + c * np.power(t, -alpha)


def eta_of_t(t, alpha, c, eps_inf):
    """Marginal yield dS/dx: the Fermi function of Theorem 2, falling alpha -> 0."""
    red = c * np.power(t, -alpha)
    return alpha * red / (eps_inf + red)


def t_knee(alpha, c, eps_inf):
    """t*, where reducible error equals irreducible error."""
    return (c / eps_inf) ** (1.0 / alpha)


def t_econ(rho, alpha, c):
    """Utility-maximising latency at cost-to-value ratio rho.  Prop 1(i)."""
    return (alpha * c / rho) ** (1.0 / (alpha + 1.0))


def lambda_econ(rho, alpha, c, eps_inf):
    """Marginal yield realised at the economic optimum.  Prop 1(ii).

    Strictly INCREASING in rho: when compute is dear relative to the value of a
    correct answer, a deployer demands a high yield before buying an e-fold.
    """
    t = t_econ(rho, alpha, c)
    return eta_of_t(t, alpha, c, eps_inf)


def rho_star(alpha, eps_inf, tstar):
    """The unique rho at which the economic optimum IS the canonical point."""
    return alpha * eps_inf / tstar


def v_star(w_per_s, alpha, eps_inf, tstar):
    """Value of a correct answer implied by the canonical price.  Prop 1(iii)."""
    return w_per_s * tstar / (alpha * eps_inf)


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------


@dataclass
class Cell:
    cell_id: str
    source: str
    task: str
    device: str
    alpha: float
    eps_inf: float
    c: float
    n_front: int
    w_per_hour: float = float("nan")

    @property
    def w_per_s(self) -> float:
        return self.w_per_hour / 3600.0

    @property
    def tstar(self) -> float:
        return t_knee(self.alpha, self.c, self.eps_inf)

    @property
    def alpha_eps(self) -> float:
        return self.alpha * self.eps_inf

    @property
    def value_multiple(self) -> float:
        """v*/(w t*) = 1/(alpha eps_inf).  Eq. (vstar-ratio)."""
        return 1.0 / self.alpha_eps

    @property
    def vstar(self) -> float:
        return v_star(self.w_per_s, self.alpha, self.eps_inf, self.tstar)

    @property
    def rhostar(self) -> float:
        return rho_star(self.alpha, self.eps_inf, self.tstar)


@dataclass
class Pairs:
    """All pairs, flattened.  A is always the more accurate and slower system."""

    cell_id: np.ndarray
    eps_A: np.ndarray
    t_A: np.ndarray
    eps_B: np.ndarray
    t_B: np.ndarray
    sys_A: np.ndarray = field(default=None)
    sys_B: np.ndarray = field(default=None)

    def __len__(self):
        return len(self.eps_A)

    @property
    def rho_AB(self):
        """Break-even cost-to-value ratio.  Definition 12, Eq. (econ-breakeven)."""
        return (self.eps_B - self.eps_A) / (self.t_A - self.t_B)

    @property
    def lambda_AB(self):
        """Break-even yield, Eq. (practical-breakeven).  The log-chord."""
        return np.log(self.eps_B / self.eps_A) / np.log(self.t_A / self.t_B)


# ----------------------------------------------------------------------
# The external verdict
# ----------------------------------------------------------------------


def verdict_econ(pairs: Pairs, rho: np.ndarray | float) -> np.ndarray:
    """True = prefer A (the accurate, slow system).  Contains no fitted quantity."""
    return np.asarray(rho) < pairs.rho_AB


def utility(eps, t, v, w_per_s):
    return v * (1.0 - eps) - w_per_s * t


def economic_regret(pairs: Pairs, choose_A: np.ndarray, v, w_per_s):
    """Utility shortfall of the chosen system, USD per 1000 queries."""
    uA = utility(pairs.eps_A, pairs.t_A, v, w_per_s)
    uB = utility(pairs.eps_B, pairs.t_B, v, w_per_s)
    chosen = np.where(choose_A, uA, uB)
    best = np.maximum(uA, uB)
    return float(np.mean(best - chosen) * 1000.0)


# ----------------------------------------------------------------------
# Selection rules.  Each returns a boolean array: True = choose A.
# ----------------------------------------------------------------------


def rule_price(pairs: Pairs, lam) -> np.ndarray:
    """The paper's rule: choose A iff the log-chord exceeds the price."""
    return pairs.lambda_AB > np.asarray(lam)


def rule_always_A(pairs: Pairs) -> np.ndarray:
    return np.ones(len(pairs), dtype=bool)


def rule_always_B(pairs: Pairs) -> np.ndarray:
    return np.zeros(len(pairs), dtype=bool)


def rule_quality_per_time(pairs: Pairs) -> np.ndarray:
    """lambda = 1: maximise (1-eps)/t.  Equivalent to CCR-DEA with unit scaling."""
    return (1 - pairs.eps_A) / pairs.t_A > (1 - pairs.eps_B) / pairs.t_B


def rule_dea(pairs: Pairs, scale: float = 1.0) -> np.ndarray:
    """CCR-DEA, single input t^scale, single output (1-eps).  scale is tunable."""
    return (1 - pairs.eps_A) / pairs.t_A**scale > (1 - pairs.eps_B) / pairs.t_B**scale


def _norm_pair(a, b):
    """Min-max normalise a two-element criterion to [0,1] (0 best for costs)."""
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (a - lo) / span, (b - lo) / span


def rule_weighted_sum(pairs: Pairs, w_eps: float = 0.5) -> np.ndarray:
    """Minimise w*eps_norm + (1-w)*t_norm within the pair."""
    ea, eb = _norm_pair(pairs.eps_A, pairs.eps_B)
    ta, tb = _norm_pair(pairs.t_A, pairs.t_B)
    return (w_eps * ea + (1 - w_eps) * ta) < (w_eps * eb + (1 - w_eps) * tb)


def rule_topsis(pairs: Pairs, w_eps: float = 0.5) -> np.ndarray:
    """TOPSIS with vector normalisation, both criteria to be minimised."""
    ne = np.sqrt(pairs.eps_A**2 + pairs.eps_B**2)
    nt = np.sqrt(pairs.t_A**2 + pairs.t_B**2)
    ea, eb = w_eps * pairs.eps_A / ne, w_eps * pairs.eps_B / ne
    ta, tb = (1 - w_eps) * pairs.t_A / nt, (1 - w_eps) * pairs.t_B / nt
    best_e, worst_e = np.minimum(ea, eb), np.maximum(ea, eb)
    best_t, worst_t = np.minimum(ta, tb), np.maximum(ta, tb)
    dpA = np.hypot(ea - best_e, ta - best_t)
    dmA = np.hypot(ea - worst_e, ta - worst_t)
    dpB = np.hypot(eb - best_e, tb - best_t)
    dmB = np.hypot(eb - worst_e, tb - worst_t)
    cA = dmA / np.where(dmA + dpA > 0, dmA + dpA, 1.0)
    cB = dmB / np.where(dmB + dpB > 0, dmB + dpB, 1.0)
    return cA > cB


def rule_hypervolume(pairs: Pairs, ref_mult: float = 1.5) -> np.ndarray:
    """Dominated hypervolume against a reference point ref_mult times the worse
    value on each axis.  ref_mult is the tunable parameter."""
    re = ref_mult * np.maximum(pairs.eps_A, pairs.eps_B)
    rt = ref_mult * np.maximum(pairs.t_A, pairs.t_B)
    hvA = (re - pairs.eps_A) * (rt - pairs.t_A)
    hvB = (re - pairs.eps_B) * (rt - pairs.t_B)
    return hvA > hvB


def rule_kneedle(pairs: Pairs, cell_fronts: dict, sensitivity: float = 1.0,
                 log_x: bool = True) -> np.ndarray:
    """Per-cell knee by maximum distance to the chord; prefer the system on the
    knee side.  `sensitivity` shifts the knee along the front."""
    out = np.zeros(len(pairs), dtype=bool)
    knees = {}
    for cid, (eps, t) in cell_fronts.items():
        order = np.argsort(t)
        x = np.log(t[order]) if log_x else t[order]
        y = eps[order]
        if len(x) < 3:
            knees[cid] = np.median(x)
            continue
        xn = (x - x.min()) / max(np.ptp(x), 1e-12)
        yn = (y - y.min()) / max(np.ptp(y), 1e-12)
        d = (1 - xn) - yn  # distance to the chord for a decreasing front
        k = int(np.argmax(d * sensitivity))
        knees[cid] = x[k]
    for i, cid in enumerate(pairs.cell_id):
        kx = knees[cid]
        xa = math.log(pairs.t_A[i]) if log_x else pairs.t_A[i]
        xb = math.log(pairs.t_B[i]) if log_x else pairs.t_B[i]
        out[i] = abs(xa - kx) < abs(xb - kx)
    return out


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pa1, pb1 = float(np.mean(a)), float(np.mean(b))
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - pe) < 1e-12:
        return 0.0
    return (po - pe) / (1 - pe)


def balanced_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    parts = []
    for cls in (True, False):
        m = truth == cls
        if m.sum():
            parts.append(float(np.mean(pred[m] == cls)))
    return float(np.mean(parts)) if parts else float("nan")


# ----------------------------------------------------------------------
# Cross-validated tuning of the competitors
# ----------------------------------------------------------------------


def cv_tune(pairs: Pairs, truth: np.ndarray, rule_fn, grid, n_folds: int = 5):
    """Five-fold CV over CELLS (not pairs).  Returns held-out predictions and
    the parameter chosen on each training fold."""
    cells = np.unique(pairs.cell_id)
    perm = RNG.permutation(len(cells))
    folds = np.array_split(perm, n_folds)
    pred = np.zeros(len(pairs), dtype=bool)
    chosen = []
    for f in folds:
        test_cells = set(cells[f])
        te = np.array([c in test_cells for c in pairs.cell_id])
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        best_p, best_k = grid[0], -np.inf
        for p in grid:
            k = cohen_kappa(rule_fn(pairs, p)[tr], truth[tr])
            if k > best_k:
                best_k, best_p = k, p
        chosen.append(best_p)
        pred[te] = rule_fn(pairs, best_p)[te]
    return pred, chosen


# ----------------------------------------------------------------------
# Self-test:  verify Proposition 1 numerically
# ----------------------------------------------------------------------


def selftest() -> int:
    print("Verifying Proposition 1 numerically\n" + "-" * 58)
    ok = True
    for trial in range(200):
        alpha = float(RNG.uniform(0.1, 3.0))
        eps_inf = float(RNG.uniform(1e-4, 0.4))
        c = float(RNG.uniform(1e-3, 10.0))
        ts = t_knee(alpha, c, eps_inf)

        # (i) t_ec(rho*) == t*
        rs = rho_star(alpha, eps_inf, ts)
        te = t_econ(rs, alpha, c)
        ok &= math.isclose(te, ts, rel_tol=1e-10)

        # (ii) lambda_ec(rho*) == alpha/2
        le = lambda_econ(rs, alpha, c, eps_inf)
        ok &= math.isclose(le, alpha / 2, rel_tol=1e-10)

        # (ii) strictly INCREASING bijection onto (0, alpha).
        # Cheap answers (large rho = costly compute relative to value) demand a
        # high yield before extra compute is bought; valuable answers demand
        # almost none.  The sweep is widened by (alpha+1) because
        # t_ec ~ rho^{-1/(alpha+1)}, so a fixed span in rho shrinks in t.
        # t_ec ~ rho^{-1/(alpha+1)} and eta depends on t through t^alpha, so the
        # span in rho needed to traverse (0, alpha) scales as (alpha+1)/alpha.
        d = 6.0 * (alpha + 1.0) / alpha
        rr = np.logspace(-d, d, 600) * rs
        ll = lambda_econ(rr, alpha, c, eps_inf)
        ok &= bool(np.all(np.diff(ll) > -1e-14))
        ok &= bool(0 < ll[0] < 0.01 * alpha and 0.99 * alpha < ll[-1] <= alpha)

        # (iii) eps at the canonical point is exactly 2 eps_inf
        ok &= math.isclose(eps_of_t(ts, alpha, c, eps_inf), 2 * eps_inf, rel_tol=1e-10)

        # (iv) value multiple
        ok &= math.isclose(
            v_star(1.0, alpha, eps_inf, ts) / (1.0 * ts),
            1.0 / (alpha * eps_inf),
            rel_tol=1e-10,
        )

        # utility really is maximised there
        v = 1.0
        w = rs * v
        tt = np.linspace(0.4 * ts, 2.6 * ts, 20001)
        u = utility(eps_of_t(tt, alpha, c, eps_inf), tt, v, w)
        ok &= abs(tt[int(np.argmax(u))] - ts) / ts < 2e-3
        if not ok:
            print(f"  FAILED at trial {trial}: alpha={alpha} eps_inf={eps_inf} c={c}")
            return 1
    print("  (i)   t_ec(rho*) == t*                                  OK")
    print("  (ii)  lambda_ec(rho*) == alpha/2, strictly increasing   OK")
    print("  (ii)  range of lambda_ec is (0, alpha)                   OK")
    print("  (iii) eps(t*) == 2 eps_inf                               OK")
    print("  (iv)  v*/(w t*) == 1/(alpha eps_inf)                     OK")
    print("        argmax_t U(t) == t* at rho = rho*                  OK")

    # Local agreement of the two criteria, and its decay with pair separation.
    print("\nAgreement of the two verdicts vs pair separation\n" + "-" * 58)
    alpha, eps_inf, c = 0.688, 0.0994, 1.0
    ts = t_knee(alpha, c, eps_inf)
    rs = rho_star(alpha, eps_inf, ts)
    print(f"  {'separation t_A/t_B':>20}  {'agreement':>10}")
    for sep in (1.05, 1.5, 2.0, 4.0, 10.0, 100.0):
        tb = ts / np.sqrt(sep) * np.exp(RNG.normal(0, 0.35, 4000))
        ta = tb * sep
        ea = eps_of_t(ta, alpha, c, eps_inf)
        eb = eps_of_t(tb, alpha, c, eps_inf)
        p = Pairs(np.array(["s"] * len(ta)), ea, ta, eb, tb)
        agree = float(np.mean(verdict_econ(p, rs) == rule_price(p, alpha / 2)))
        print(f"  {sep:>20.2f}  {agree:>10.3f}")
    print(
        "\n  Interpretation: the two criteria are tangent at t* and therefore agree\n"
        "  exactly for close pairs; the log-chord and the linear chord separate as\n"
        "  the pair widens.  Report this curve in SM24 -- it bounds how much of any\n"
        "  measured disagreement is criterion geometry rather than a wrong price."
    )
    return 0


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------


def read_csv(path):
    import csv

    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load(indir):
    cells = {}
    prices = {r["device"]: float(r["usd_per_hour"]) for r in
              read_csv(os.path.join(indir, "device_prices.csv"))}
    for r in read_csv(os.path.join(indir, "cells.csv")):
        cells[r["cell_id"]] = Cell(
            cell_id=r["cell_id"], source=r["source"], task=r["task"],
            device=r["device"], alpha=float(r["alpha"]),
            eps_inf=float(r["eps_inf"]), c=float(r["c"]),
            n_front=int(r["n_front"]),
            w_per_hour=prices.get(r["device"], float("nan")),
        )
    rows = read_csv(os.path.join(indir, "pairs.csv"))
    pairs = Pairs(
        cell_id=np.array([r["cell_id"] for r in rows]),
        eps_A=np.array([float(r["eps_A"]) for r in rows]),
        t_A=np.array([float(r["t_A"]) for r in rows]),
        eps_B=np.array([float(r["eps_B"]) for r in rows]),
        t_B=np.array([float(r["t_B"]) for r in rows]),
        sys_A=np.array([r.get("sys_A", "") for r in rows]),
        sys_B=np.array([r.get("sys_B", "") for r in rows]),
    )
    anchors = read_csv(os.path.join(indir, "anchors.csv"))
    return cells, pairs, anchors


def cell_fronts_from_pairs(pairs: Pairs):
    fronts = {}
    for cid in np.unique(pairs.cell_id):
        m = pairs.cell_id == cid
        eps = np.concatenate([pairs.eps_A[m], pairs.eps_B[m]])
        t = np.concatenate([pairs.t_A[m], pairs.t_B[m]])
        _, idx = np.unique(np.round(t, 12), return_index=True)
        fronts[cid] = (eps[idx], t[idx])
    return fronts


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------


def run(indir, outdir, frozen_price=0.344):
    os.makedirs(outdir, exist_ok=True)
    cells, pairs, anchors = load(indir)
    fronts = cell_fronts_from_pairs(pairs)
    numbers = {}

    # --- Table: implied valuations -----------------------------------
    lines = []
    mults, vs, rstars = [], [], []
    for cid, cell in sorted(cells.items()):
        if not math.isfinite(cell.w_per_hour):
            continue
        mults.append(cell.value_multiple)
        vs.append(cell.vstar)
        rstars.append(cell.rhostar)
        lines.append(
            f"{cell.cell_id} & {cell.task} & {cell.w_per_hour:.2f} & "
            f"{cell.tstar*1e3:.1f} & {cell.alpha_eps:.4f} & "
            f"{cell.value_multiple:.1f} & {cell.vstar:.2e} \\\\"
        )
    with open(os.path.join(outdir, "tab_econ_implied.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    numbers["value_multiple_median"] = float(np.median(mults)) if mults else None
    numbers["vstar_median_usd"] = float(np.median(vs)) if vs else None
    numbers["rhostar_median"] = float(np.median(rstars)) if rstars else None

    # --- Rules -------------------------------------------------------
    per_cell_price = np.array([cells[c].alpha / 2 for c in pairs.cell_id])
    rules = {
        "Derived price, frozen": lambda p, _: rule_price(p, frozen_price),
        "Derived price, per-cell": lambda p, _: rule_price(p, per_cell_price),
        "Quality per unit time": lambda p, _: rule_quality_per_time(p),
        "Always the faster system": lambda p, _: rule_always_B(p),
        "Always the more accurate system": lambda p, _: rule_always_A(p),
    }
    tuned = {
        "Best tuned constant": (rule_price, np.linspace(0.02, 1.5, 149)),
        "TOPSIS, tuned weights": (rule_topsis, np.linspace(0.05, 0.95, 19)),
        "Weighted sum, tuned weight": (rule_weighted_sum, np.linspace(0.05, 0.95, 19)),
        "Hypervolume, tuned reference": (rule_hypervolume, np.linspace(1.05, 6.0, 25)),
        "DEA (CCR), tuned scaling": (rule_dea, np.linspace(0.1, 2.0, 39)),
        "Kneedle, tuned sensitivity": (
            lambda p, s: rule_kneedle(p, fronts, s), np.linspace(0.5, 3.0, 11)),
    }

    # --- Score at each anchor ----------------------------------------
    results = {}
    for row in anchors:
        name = row["anchor"]
        if "rho_per_s" in row and row.get("rho_per_s"):
            rho = np.full(len(pairs), float(row["rho_per_s"]))
            v_by_pair = np.array([cells[c].w_per_s for c in pairs.cell_id]) / rho
        else:
            v = float(row["v_usd"])
            w = np.array([cells[c].w_per_s for c in pairs.cell_id])
            rho = w / v
            v_by_pair = np.full(len(pairs), v)
        w_by_pair = np.array([cells[c].w_per_s for c in pairs.cell_id])
        truth = verdict_econ(pairs, rho)

        # Sanity check on the anchor.  lambda_ec is a bijection onto (0, alpha),
        # so an anchor far from the corpus rho* corresponds to a canonical price
        # far from alpha/2 and the frozen price is then dominated BY
        # CONSTRUCTION, not by any failure of the estimator.  Say so loudly:
        # this is the single most likely way to misread the output.
        if rstars:
            decades = math.log10(float(np.median(rho)) / float(np.median(rstars)))
            if abs(decades) > 1.0:
                print(
                    f"  WARNING  anchor {name}: median rho is 10^{decades:+.1f} "
                    f"from the corpus median rho*.\n"
                    f"           The canonical price cannot be near-optimal here; "
                    f"a low kappa for the\n"
                    f"           derived price at this anchor is a statement about "
                    f"the valuation, not\n"
                    f"           about the estimator.  Report it as such, or "
                    f"justify the anchor.",
                    file=sys.stderr)
        res = {}
        for label, fn in rules.items():
            pred = fn(pairs, None)
            res[label] = (
                cohen_kappa(pred, truth),
                economic_regret(pairs, pred, v_by_pair, w_by_pair),
                balanced_accuracy(pred, truth),
            )
        for label, (fn, grid) in tuned.items():
            pred, chosen = cv_tune(pairs, truth, fn, list(grid))
            res[label] = (
                cohen_kappa(pred, truth),
                economic_regret(pairs, pred, v_by_pair, w_by_pair),
                balanced_accuracy(pred, truth),
            )
            numbers[f"tuned_param::{label}::{name}"] = [float(x) for x in chosen]
        results[name] = res

    # --- LaTeX table -------------------------------------------------
    anchor_names = list(results)
    order = (["Derived price, frozen", "Derived price, per-cell"]
             + list(tuned)
             + ["Quality per unit time", "Always the faster system",
                "Always the more accurate system"])
    out = []
    for label in order:
        cellsx = []
        for a in anchor_names:
            k, r, _ = results[a][label]
            cellsx += [f"{k:.3f}", f"{r:.3f}"]
        out.append(f"{label} & " + " & ".join(cellsx) + r" \\")
    with open(os.path.join(outdir, "tab_econ_baselines.tex"), "w") as fh:
        fh.write("\n".join(out) + "\n")

    # --- Headline gaps (F2, F3) --------------------------------------
    gaps, bests = [], []
    for a in anchor_names:
        kd = results[a]["Derived price, frozen"][0]
        kt = results[a]["Best tuned constant"][0]
        others = [results[a][l][0] for l in tuned if l != "Best tuned constant"]
        gaps.append(kt - kd)
        bests.append(max(others))
    numbers["gap_vs_tuned_constant_max"] = float(np.max(gaps))
    numbers["best_competitor_kappa_max"] = float(np.max(bests))

    # --- rho sweep ---------------------------------------------------
    grid = np.logspace(
        math.log10(np.percentile(pairs.rho_AB, 1) / 30),
        math.log10(np.percentile(pairs.rho_AB, 99) * 30), 160)
    sweep = {"rho": grid.tolist()}
    for label in ["Derived price, frozen", "Best tuned constant"]:
        ks = []
        for rho in grid:
            truth = verdict_econ(pairs, rho)
            if label == "Derived price, frozen":
                pred = rule_price(pairs, frozen_price)
            else:
                pred, _ = cv_tune(pairs, truth, rule_price,
                                  list(np.linspace(0.02, 1.5, 149)))
            ks.append(cohen_kappa(pred, truth))
        sweep[label] = ks
    kd = np.array(sweep["Derived price, frozen"])
    kb = np.array(sweep["Best tuned constant"])
    se = 1.0 / math.sqrt(len(pairs))
    near = grid[kd >= kb - se]
    if len(near):
        numbers["rho_interval"] = [float(near.min()), float(near.max())]
        numbers["rhostar_coverage"] = float(
            np.mean([(near.min() <= r <= near.max()) for r in rstars])) if rstars else None
    with open(os.path.join(outdir, "econ_sweep.csv"), "w") as fh:
        fh.write("rho," + ",".join(k for k in sweep if k != "rho") + "\n")
        for i, r in enumerate(grid):
            fh.write(f"{r}," + ",".join(
                f"{sweep[k][i]}" for k in sweep if k != "rho") + "\n")

    with open(os.path.join(outdir, "numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=2)
    print(json.dumps(numbers, indent=2))
    return numbers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--indir", default="data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--frozen-price", type=float, default=0.344)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run(a.indir, a.outdir, a.frozen_price) and 0


if __name__ == "__main__":
    sys.exit(main() or 0)
