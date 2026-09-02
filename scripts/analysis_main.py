"""
Master analysis for the elasticity-knee theory.
Produces: results.json, table_cells.csv, table_rules.csv  and all figures.
"""
import csv, math, json, itertools, collections, os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, wilcoxon

RNG = np.random.default_rng(20260824)
# resolve data/output locations: works both in the artifacts tree and flat
_D = os.environ.get("DATA", "data")
_OUT = os.environ.get("OUT", "results")
D, OUT = _D, _OUT
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Build cells:  (family, cell, task)  ->  list of (t, eps)
# ----------------------------------------------------------------------
cells = collections.defaultdict(list)

# --- inference: timm + ultralytics -----------------------------------
for x in csv.DictReader(open(f'{D}/corpus.csv')):
    Q, ch, t = float(x['Q']), float(x['chance']), float(x['latency_ms'])
    if t <= 0: continue
    e = 1 - (Q - ch) / (1 - ch)
    if 0 < e < 1:
        cells[(x['source'], x['cell'], x['task'])].append((t, e))

# --- MAI challenges: PSNR -> normalised MSE distortion ---------------
for x in csv.DictReader(open(f'{D}/mai_data.csv')):
    t = float(x['runtime_ms']); p = float(x['psnr'])
    if t <= 0: continue
    e = 10 ** (-p / 10.0)          # MSE / peak^2 , a genuine distortion
    cells[('mai', x['cell'], 'Super-resolution')].append((t, e))


def pareto(pts):
    pts = sorted(pts); out = []; best = np.inf
    for t, e in pts:
        if e < best - 1e-15:
            out.append((t, e)); best = e
    return out


def fit_knee(t, e):
    """eps(t) = eps_inf + c t^-alpha  ->  (eps_inf, c, alpha, R2)"""
    lr = np.log(t); ee = np.array(e)
    f = lambda lr, ei, lc, a: ei + np.exp(lc) * np.exp(-a * lr)
    best = None
    for ei0 in (0.0, .3, .6, .9):
        try:
            p, _ = curve_fit(f, lr, ee, p0=[min(ee) * ei0, 0., .5], maxfev=400000,
                             bounds=([0, -40, .01], [min(ee) * .999, 40, 8]))
        except Exception:
            continue
        r = ee - f(lr, *p); ss = 1 - (r ** 2).sum() / ((ee - ee.mean()) ** 2).sum()
        if best is None or ss > best[3]:
            best = (p[0], math.exp(p[1]), p[2], ss)
    return best


MINFRONT = 6
ref = {}
for k, v in cells.items():
    P = pareto(v)
    if len(P) < MINFRONT: continue
    fit = fit_knee([p[0] for p in P], [p[1] for p in P])
    if fit is None: continue
    ei, c, a, r2 = fit
    if not (0 < a < 8): continue
    if ei <= 1e-6 or ei >= 0.999 * min(p[1] for p in P): continue   # floor at bound = not identified
    ref[k] = dict(P=P, n=len(v), eps_inf=ei, c=c, alpha=a, r2=r2, lam=a / 2,
                  t_knee=(c / ei) ** (1 / a) if ei > 0 else np.nan)

print(f'cells fitted: {len(ref)}')

# ----------------------------------------------------------------------
# 2. Per-cell table
# ----------------------------------------------------------------------
with open(f'{OUT}/table_cells.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['family', 'cell', 'task', 'n', 'n_front', 'eps_inf', 'alpha',
                'lambda_star', 'R2', 't_knee_ms', 't_min', 't_max'])
    for k, d in sorted(ref.items()):
        w.writerow([k[0], k[1], k[2], d['n'], len(d['P']),
                    f"{d['eps_inf']:.4f}", f"{d['alpha']:.4f}", f"{d['lam']:.4f}",
                    f"{d['r2']:.4f}", f"{d['t_knee']:.4g}",
                    f"{d['P'][0][0]:.4g}", f"{d['P'][-1][0]:.4g}"])

lams = np.array([d['lam'] for d in ref.values()])
alphas = np.array([d['alpha'] for d in ref.values()])
bs = np.array([np.median(RNG.choice(lams, len(lams))) for _ in range(20000)])
LAM_HAT = float(np.median(lams))
CI = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
print(f'lambda* median = {LAM_HAT:.3f}  CI {CI}  range [{lams.min():.3f},{lams.max():.3f}] CV={lams.std()/lams.mean():.3f}')

# ----------------------------------------------------------------------
# 3. Pair-level verification
# ----------------------------------------------------------------------
SEP = math.log(1.5)
pairs = []
for k, d in ref.items():
    for (tf, ef), (ts, es) in itertools.combinations(d['P'], 2):
        if ts <= tf or es >= ef: continue
        dlr = math.log(ts / tf)
        if dlr < SEP: continue
        lab = math.log(ef / es) / dlr
        pairs.append(dict(k=k, tf=tf, ef=ef, ts=ts, es=es, dlr=dlr, lab=lab,
                          truth='slow' if lab > d['lam'] else 'fast'))
print(f'pairs = {len(pairs)}')

loco = {k: float(np.median([d['lam'] for kk, d in ref.items() if kk != k])) for k in ref}
loto = {}
for k in ref:
    o = [d['lam'] for kk, d in ref.items() if kk[2] != k[2]]
    loto[k] = float(np.median(o)) if o else LAM_HAT
lofam = {}
for k in ref:
    o = [d['lam'] for kk, d in ref.items() if kk[0] != k[0]]
    lofam[k] = float(np.median(o)) if o else LAM_HAT


def score(fn):
    ok = und = 0; reg = 0.0
    for p in pairs:
        v = fn(p)
        if v is None: und += 1; continue
        if v == p['truth']: ok += 1
        else: reg += abs(ref[p['k']]['lam'] - p['lab']) * p['dlr']
    dec = len(pairs) - und
    return dict(acc=100 * ok / dec if dec else 0.0,
                undef=100 * und / len(pairs), regret=reg / len(pairs))


price = lambda lam: (lambda p: 'slow' if p['lab'] > lam else 'fast')
rules = collections.OrderedDict()
rules['Universal knee price $\\lambda=0.35$'] = price(0.35)
rules['Leave-one-cell-out knee price'] = lambda p: 'slow' if p['lab'] > loco[p['k']] else 'fast'
rules['Leave-one-task-out knee price'] = lambda p: 'slow' if p['lab'] > loto[p['k']] else 'fast'
rules['Leave-one-family-out knee price'] = lambda p: 'slow' if p['lab'] > lofam[p['k']] else 'fast'
for L in (0.166, 0.2, 0.25, 0.30, 0.40, 0.50, 1.0):
    rules[f'Fixed $\\lambda={L}$'] = price(L)
rules['Accuracy per unit time ($\\varepsilon t$)'] = price(1.0)
rules['Always the faster system'] = lambda p: 'fast'
rules['Always the more accurate system'] = lambda p: 'slow'
rules['Coin flip'] = lambda p: 'slow' if RNG.random() < .5 else 'fast'

res_rules = {n: score(f) for n, f in rules.items()}
with open(f'{OUT}/table_rules.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['rule', 'accuracy_pct', 'undefined_pct', 'mean_regret_nats'])
    for n, r in res_rules.items():
        w.writerow([n, f"{r['acc']:.2f}", f"{r['undef']:.2f}", f"{r['regret']:.4f}"])
        print(f'{n:44s} {r["acc"]:6.2f}  {r["regret"]:.4f}')

# agreement statistics between full and simplified theory
V = np.array([p['truth'] == 'slow' for p in pairs])
W = np.array([p['lab'] > 0.35 for p in pairs])
po = float((V == W).mean())
pe = float(V.mean() * W.mean() + (1 - V.mean()) * (1 - W.mean()))
kappa = (po - pe) / (1 - pe)
phi = float(np.corrcoef(V, W)[0, 1])

# per-cell sigma-ranking correlation and single-winner agreement
rho, winner = [], []
for k, d in ref.items():
    P = d['P']
    sf = [-math.log(e) - d['lam'] * math.log(t) for t, e in P]
    ss = [-math.log(e) - 0.35 * math.log(t) for t, e in P]
    rho.append(spearmanr(sf, ss).statistic)
    winner.append(int(np.argmax(sf)) == int(np.argmax(ss)))

# n-way selection
nway = {}
for K in (2, 3, 5, 10, 20):
    hit = tot = 0
    for k, d in ref.items():
        P = d['P']
        if len(P) < K: continue
        for _ in range(500):
            idx = RNG.choice(len(P), K, replace=False); C = [P[i] for i in idx]
            am = lambda lam: max(range(K), key=lambda i: -math.log(C[i][1]) - lam * math.log(C[i][0]))
            tot += 1; hit += (am(0.35) == am(d['lam']))
    nway[K] = dict(n=tot, acc=100 * hit / tot if tot else 0.0)

# robustness of the constant: sweep
sweep = {f'{L:.2f}': score(price(L))['acc'] for L in np.arange(0.05, 1.01, 0.05)}

# measurement-noise stress: perturb eps by binomial noise of a 50k test set
stress = {}
for n_test in (1000, 5000, 50000):
    ok = tot = 0
    for p in pairs:
        ef = RNG.binomial(n_test, min(p['ef'], .999)) / n_test
        es = RNG.binomial(n_test, min(p['es'], .999)) / n_test
        if ef <= 0 or es <= 0 or ef <= es: continue
        lab = math.log(ef / es) / p['dlr']
        tot += 1
        ok += (('slow' if lab > 0.35 else 'fast') == p['truth'])
    stress[n_test] = 100 * ok / tot if tot else 0.0

results = dict(
    n_cells=len(ref), n_pairs=len(pairs),
    lambda_hat=LAM_HAT, ci=CI, lam_min=float(lams.min()), lam_max=float(lams.max()),
    lam_cv=float(lams.std() / lams.mean()), alpha_median=float(np.median(alphas)),
    agreement=100 * po, kappa=kappa, phi=phi,
    rho_median=float(np.median(rho)), rho_min=float(np.min(rho)),
    winner_hits=int(sum(winner)), winner_n=len(winner),
    nway=nway, rules={n: r for n, r in res_rules.items()},
    sweep=sweep, stress=stress,
    r2_median=float(np.median([d['r2'] for d in ref.values()])),
)
json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in results.items() if k not in ('rules', 'sweep')}, indent=1))
