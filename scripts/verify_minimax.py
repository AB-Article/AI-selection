#!/usr/bin/env python3
"""
verify_minimax.py  --  numerical verification of Theorem 4(iii)-(iv) in the
ex-ante form, where the realised price is an unknown state of the deployment
environment rather than the analyst's own preference.

Checks, for the binary regret R(p||q) = KL(p||q) of Theorem 5:

  V1  R is convex in p, so sup over any admissible interval is at an endpoint.
  V2  Equalizer property: for an admissible interval [p_lo, p_hi] with an
      interior minimax point, R(p_lo||q*) = R(p_hi||q*).
  V3  Symmetry: whenever [p_lo, p_hi] is symmetric about 1/2 -- that is,
      invariant under the yield reflection lambda <-> alpha - lambda -- the
      minimax declared price is exactly 1/2, for every width, not only in the
      limit of complete ignorance.
  V4  Closed form of the minimax value under symmetry:
      sup_p R(p||1/2) = ln 2 - H(p_lo), which tends to ln 2 = 1 bit as the
      admissible set fills (0,1).
  V5  Bayes: the optimal declared price is the prior mean, hence 1/2 for any
      prior symmetric about 1/2, including priors truncated to [p_lo, p_hi].
  V6  Sensitivity: how far the minimax price moves when the admissible set is
      NOT symmetric, which is the assumption the theorem rests on.

Writes minimax_verification.json
"""
import json
import os
import math

DATA = os.environ.get("DATA", "data")
OUT = os.environ.get("OUT", "results")
os.makedirs(OUT, exist_ok=True)

import numpy as np
from scipy.stats import beta as betadist

RNG = np.random.default_rng(20260824)
GRID = 2_000_001


def H(p):
    p = np.clip(p, 1e-300, 1 - 1e-300)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def KL(p, q):
    p = np.clip(np.asarray(p, float), 1e-300, 1 - 1e-300)
    q = np.clip(np.asarray(q, float), 1e-300, 1 - 1e-300)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def minimax(plo, phi, n=GRID):
    """argmin_q max_{p in [plo,phi]} KL(p||q), using convexity of R in p."""
    q = np.linspace(1e-9, 1 - 1e-9, n)
    worst = np.maximum(KL(plo, q), KL(phi, q))
    i = int(np.argmin(worst))
    return float(q[i]), float(worst[i])


res = {}

# -- V1 convexity of the regret in the truth ---------------------------------
worst_dev = 0.0
for q0 in (0.05, 0.2, 0.5, 0.8, 0.95):
    p = np.linspace(1e-6, 1 - 1e-6, 200001)
    r = KL(p, q0)
    second = np.diff(r, 2)
    worst_dev = min(worst_dev, float(second.min()))
res["convexity_min_second_difference"] = worst_dev
print(f"V1 convexity in p: min second difference {worst_dev:.3e} (must be >= 0)")

# -- V2/V3 symmetric admissible sets -----------------------------------------
sym_rows, sym_err = [], 0.0
for d in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49, 0.499):
    plo, phi = 0.5 - d, 0.5 + d
    q, v = minimax(plo, phi)
    closed = math.log(2) - float(H(plo))
    sym_err = max(sym_err, abs(q - 0.5))
    sym_rows.append(dict(half_width=d, q_star=q, value=v, closed_form=closed,
                         value_err=abs(v - closed)))
    print(f"V3 [{plo:.3f},{phi:.3f}]  q*={q:.9f}  value={v:.6f}  "
          f"ln2-H(p_lo)={closed:.6f}")
res["symmetric"] = sym_rows
res["symmetric_max_dev_from_half"] = sym_err
res["symmetric_max_value_err"] = max(r["value_err"] for r in sym_rows)
print(f"V3 max |q*-1/2| over symmetric sets: {sym_err:.2e}")
print(f"V4 max |value - (ln2 - H(p_lo))|:    "
      f"{res['symmetric_max_value_err']:.2e}")

# equalizer check at the symmetric optimum
eq = max(abs(float(KL(0.5 - d, 0.5)) - float(KL(0.5 + d, 0.5)))
         for d in (0.1, 0.2, 0.3, 0.4))
res["equalizer_symmetric"] = eq
print(f"V2 equalizer at q*=1/2, symmetric sets: max gap {eq:.2e}")

# -- V2 equalizer on asymmetric sets -----------------------------------------
asym_rows = []
for plo, phi in ((0.2, 0.9), (0.3, 0.9), (0.1, 0.6), (0.25, 0.55),
                 (0.05, 0.45), (0.4, 0.95)):
    q, v = minimax(plo, phi)
    gap = abs(float(KL(plo, q)) - float(KL(phi, q)))
    asym_rows.append(dict(p_lo=plo, p_hi=phi, q_star=q, value=v,
                          equalizer_gap=gap, shift=abs(q - 0.5)))
    print(f"V6 [{plo:.2f},{phi:.2f}]  q*={q:.6f}  shift={abs(q-0.5):.4f}  "
          f"equalizer gap={gap:.2e}")
res["asymmetric"] = asym_rows
res["asymmetric_max_equalizer_gap"] = max(r["equalizer_gap"] for r in asym_rows)
res["asymmetric_max_shift"] = max(r["shift"] for r in asym_rows)

# -- V5 Bayes, including truncated priors ------------------------------------
qg = np.linspace(1e-4, 1 - 1e-4, 200001)
dev = 0.0
for _ in range(300):
    a_, b_ = float(RNG.uniform(0.3, 8)), float(RNG.uniform(0.3, 8))
    ps = np.clip(betadist.rvs(a_, b_, size=4000,
                              random_state=int(RNG.integers(1 << 31))),
                 1e-6, 1 - 1e-6)
    mp = float(ps.mean())
    obj = -mp * np.log(qg) - (1 - mp) * np.log(1 - qg)
    dev = max(dev, abs(float(qg[int(np.argmin(obj))]) - mp))
res["bayes_max_dev_from_prior_mean"] = dev
print(f"V5 Bayes, 300 Beta priors: max |argmin - prior mean| = {dev:.2e}")

trunc = 0.0
for d in (0.1, 0.2, 0.3, 0.4):
    for a_ in (0.5, 1.0, 3.0):
        p = np.linspace(0.5 - d, 0.5 + d, 60001)
        w = (p ** (a_ - 1)) * ((1 - p) ** (a_ - 1))      # symmetric Beta
        w /= w.sum()
        mp = float((w * p).sum())
        obj = -mp * np.log(qg) - (1 - mp) * np.log(1 - qg)
        trunc = max(trunc, abs(float(qg[int(np.argmin(obj))]) - 0.5))
res["bayes_truncated_symmetric_max_dev"] = trunc
print(f"V5 Bayes, symmetric priors truncated to [1/2-d,1/2+d]: "
      f"max |q*-1/2| = {trunc:.2e}")

json.dump(res, open(f"{OUT}/minimax_verification.json", "w"), indent=1)
print("\nwrote minimax_verification.json")
