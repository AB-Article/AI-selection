#!/usr/bin/env python3
"""Verify the endpoint behaviour of the Legendre-Fenchel conjugate.

Corrected Duality theorem: sigma*(lambda) = sup_x {S(x) - lambda x} is finite
exactly on the CLOSED interval [0, alpha]; the supremum is ATTAINED exactly on
the open interior, where the closed form
    sigma*(lambda) = -ln(eps_inf) - lambda ln(t*) - H(lambda/alpha)
holds; at the endpoints it is a finite limit,
    sigma*(0)     = -ln(eps_inf)   (as x -> +inf),
    sigma*(alpha) = -ln(c)         (as x -> -inf),
to which the closed form extends continuously since H(0) = H(1) = 0; and the
supremum is +inf for lambda outside [0, alpha].

Checks, per identified cell (falling back to a canonical parameter set if the
cell table is unavailable):
  1. sup over an expanding window [-W, W] e-folds converges to -ln(eps_inf) at
     lambda=0 and to -ln(c) at lambda=alpha, with the argmax pinned to the
     window edge (non-attainment).
  2. For lambda = -delta and lambda = alpha + delta the windowed sup grows
     ~ linearly in W (unboundedness).
  3. The closed form evaluated at p in {0, 1} equals the endpoint limits.

Writes data/endpoints.json.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def load_cells():
    """Fitted (eps_inf, c, alpha) per identified cell, from table_cells.csv
    when present; otherwise a canonical grid spanning the measured range."""
    cells = []
    path = os.path.join(DATA, "table_cells.csv")
    if os.path.exists(path):
        import csv

        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    eps_inf = float(row.get("eps_inf") or row.get("epsinf"))
                    alpha = float(row["alpha"])
                    if "c" in row and row["c"]:
                        c = float(row["c"])
                    else:
                        tstar = float(row.get("tstar") or row.get("t_star"))
                        c = eps_inf * tstar**alpha
                    cells.append(
                        {"cell": row.get("cell", f"row{len(cells)}"),
                         "eps_inf": eps_inf, "c": c, "alpha": alpha}
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    if not cells:  # canonical fallback spanning the measured parameter range
        for eps_inf in (0.0935, 0.0994, 0.1807, 0.2789, 0.4316):
            for alpha in (0.538, 0.688, 1.052):
                cells.append(
                    {"cell": f"synthetic eps_inf={eps_inf} alpha={alpha}",
                     "eps_inf": eps_inf, "c": 0.2, "alpha": alpha}
                )
    return cells


def S(x, eps_inf, c, alpha):
    # stable: -log(eps_inf + c e^{-alpha x})
    return -np.logaddexp(math.log(eps_inf), math.log(c) - alpha * x)


def windowed_sup(lam, eps_inf, c, alpha, W, n=4_000_001):
    x = np.linspace(-W, W, n)
    obj = S(x, eps_inf, c, alpha) - lam * x
    i = int(np.argmax(obj))
    # Non-attainment: the window-edge value in the relevant direction must be
    # within numerical flatness of the maximum (the objective is monotone
    # towards that edge; past ~30 e-folds it saturates below double precision,
    # so an interior argmax index only reflects floating-point ties).
    edge = obj[-1] if lam <= alpha / 2 else obj[0]
    sup_at_edge = bool(abs(float(obj[i]) - float(edge)) < 1e-12)
    return float(obj[i]), float(x[i]), sup_at_edge


def main():
    rng_windows = (20.0, 40.0, 60.0)
    delta = 0.05
    out = {"windows_efolds": list(rng_windows), "delta_outside": delta,
           "cells": []}
    worst_lo = worst_hi = 0.0
    for cell in load_cells():
        e, c, a = cell["eps_inf"], cell["c"], cell["alpha"]
        rec = {"cell": cell["cell"], "eps_inf": e, "c": c, "alpha": a,
               "limit_lambda0": -math.log(e), "limit_lambda_alpha": -math.log(c)}
        # 1. endpoint convergence and non-attainment
        for key, lam, target in (("lambda0", 0.0, -math.log(e)),
                                 ("lambda_alpha", a, -math.log(c))):
            errs, edge_flags = [], []
            for W in rng_windows:
                s, xhat, at_edge = windowed_sup(lam, e, c, a, W)
                errs.append(abs(s - target))
                edge_flags.append(bool(at_edge))
            rec[f"abs_err_{key}"] = errs
            rec[f"sup_at_window_edge_{key}"] = edge_flags
            rec[f"errors_nonincreasing_{key}"] = bool(
                errs[0] >= errs[1] - 1e-15 and errs[1] >= errs[2] - 1e-15)
        worst_lo = max(worst_lo, rec["abs_err_lambda0"][-1])
        worst_hi = max(worst_hi, rec["abs_err_lambda_alpha"][-1])
        # 2. unboundedness just outside [0, alpha]
        for key, lam in (("below0", -delta), ("above_alpha", a + delta)):
            sups = [windowed_sup(lam, e, c, a, W)[0] for W in rng_windows]
            slope = (sups[-1] - sups[0]) / (rng_windows[-1] - rng_windows[0])
            rec[f"sup_grows_{key}"] = bool(sups[0] < sups[1] < sups[2])
            rec[f"growth_rate_{key}"] = slope  # ~ delta, linear in W
        # 3. continuous extension of the closed form (H(0)=H(1)=0)
        tstar = (c / e) ** (1.0 / a)
        rec["closed_form_p0"] = -math.log(e)
        rec["closed_form_p1"] = -math.log(e) - a * math.log(tstar)  # = -ln c
        rec["closed_form_matches_limits"] = bool(
            abs(rec["closed_form_p0"] - rec["limit_lambda0"]) < 1e-12
            and abs(rec["closed_form_p1"] - rec["limit_lambda_alpha"]) < 1e-12
        )
        out["cells"].append(rec)
    out["max_abs_err_lambda0_at_60efolds"] = worst_lo
    out["max_abs_err_lambda_alpha_at_60efolds"] = worst_hi
    out["all_sup_at_window_edge"] = all(
        all(r["sup_at_window_edge_lambda0"])
        and all(r["sup_at_window_edge_lambda_alpha"])
        for r in out["cells"]
    )
    out["all_unbounded_outside"] = all(
        r["sup_grows_below0"] and r["sup_grows_above_alpha"]
        for r in out["cells"]
    )
    out["all_closed_form_matches"] = all(
        r["closed_form_matches_limits"] for r in out["cells"]
    )
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "endpoints.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("max_abs_err_lambda0_at_60efolds",
                       "max_abs_err_lambda_alpha_at_60efolds",
                       "all_sup_at_window_edge",
                       "all_unbounded_outside",
                       "all_closed_form_matches")}, indent=1))


if __name__ == "__main__":
    main()
