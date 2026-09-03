# The price of computation — reproducibility archive

Data and analysis code for:

> D. Ignatov, *The price of computation: a saturating frontier fixes 
> its own exchange rate in AI model selection*. Submitted to **Information Sciences**
> (Elsevier).

Everything reported in the article and its supplementary material is produced by
the code in this archive from the data in this archive. There are no hidden
steps, no manual edits to any number, and no network access at analysis time.

**Archive DOI:** `10.5281/zenodo.22261267`

---

## Quick start

```bash
python3 -m pip install -r requirements.txt
./run_all.sh fast      # ~8 minutes: everything except the slow simulations
./run_all.sh           # ~30 minutes: adds the robustness simulations
```

Outputs land in `results/` (numbers and tables) and `figures/` (PDFs). Nothing
outside those two directories is written, so a run can always be checked by
diffing them against the archived copies.

The pipeline needs only `numpy`, `scipy` and `matplotlib`. Every random seed is
fixed at `20260824`, so results are bit-reproducible on a given platform.

---

## What reproduces, and how exactly

Verified on Python 3.12 / numpy 2.4.4 / scipy 1.17.1 / matplotlib 3.10.8:

| Claim in the article | Value | Reproduced by |
|---|---|---|
| Pooled canonical price | 0.3542 | `stats_meta.py` |
| 95% confidence interval | [0.3246, 0.3837] | `stats_meta.py` |
| 95% prediction interval | [0.2626, 0.4457] | `stats_meta.py` |
| Median per-cell price | 0.3439 | `analysis_main.py` |
| Agreement of the rule at λ̂ = 0.35 | 94.98% | `analysis_main.py` |
| Cohen's κ against the fitted frontier | 0.8645 | `analysis_main.py` |
| Mean regret of the rule | 0.0040 nats | `analysis_main.py` |
| Mean regret of "always more accurate" | 0.2952 nats | `analysis_main.py` |
| Saturating law preferred by AICc | 12 of 12 cells, median Δ = 39.9 | `stats_meta.py` |
| Theorem 11(iii)–(iv), numerically | max deviation 2.5 × 10⁻⁶ | `verify_minimax.py` |
| Proposition 22 identities | exact, to machine precision | `external_criterion.py --selftest` |

Nine of the ten figures in the article regenerate **byte-identical** to the
submitted versions; `fig9_extended.pdf` differs by three bytes of embedded PDF
metadata and is visually identical.

---

## Layout

```
.
├── run_all.sh              driver: reproduces every number and figure
├── requirements.txt        pinned dependency floor
├── LICENSE                 MIT (code)
├── LICENSE-DATA            CC BY 4.0 (data)
├── CITATION.cff            machine-readable citation metadata
├── data/                   inputs (see "Data provenance" below)
├── scripts/                analysis: reads data/, writes results/
├── build/                  corpus construction from public sources
├── results/                generated numbers, tables and run logs
└── figures/                generated figures
```

`scripts/` is what a reviewer needs. `build/` documents how `data/` was
assembled from public sources; it requires network access and is not part of
`run_all.sh`.

---

## Data provenance

Every source is public and none is credentialed. Files were retrieved on
**29 August 2026**.

### Raw corpora

| File | Rows | Contents | Upstream source |
|---|---|---|---|
| `corpus.csv` | 10,504 | Primary corpus (group P): system–device inference records, ImageNet and COCO | `timm` results archive; Ultralytics model documentation |
| `corpus_extended.csv` | 3,705 | Out-of-domain corpus (group O): segmentation, detection, speech | MMSegmentation, MMDetection, Open ASR Leaderboard |
| `corpus_llm.csv` | 34,089 | Language-model latency ladders (group L) | LLM-Perf leaderboard, Open LLM Leaderboard v2, LLM-Inference-Bench |
| `corpus_retired.csv` | 7,958 | Retired-benchmark positive control (group R): Pythia and BLOOM checkpoints | Pythia suite, BLOOM, `lm-evaluation-harness` |
| `corpus_qwen.csv` | 1,805 | Qwen3 throughput/latency ladder | Qwen3 technical report |
| `mai_data.csv` | 67 | Mobile AI challenge entries, PSNR against runtime | Mobile AI & AIM 2022 challenge report |
| `bench360_table3.csv` | 28 | Local LLM inference, quality against time per output token | Bench360 |

### Derived inputs

These are outputs of `build/` scripts whose raw inputs are large leaderboard
snapshots that are not redistributed here. They are supplied so the analysis
runs end to end without network access; the builders that produce them are in
`build/` and document their own inputs.

| File | Contents | Producer |
|---|---|---|
| `pareto_slopes.csv` | 9 mobile/desktop device cells, fitted yields | `build/build_timm.py`, `build/build_ultra.py` |
| `summary_v2.json` | 73,136 training curves, per-task exponents | training-curve extraction |
| `table_ladders.csv` | 30 language-model capacity ladders, fitted | `build/llm_ladders.py` |
| `stats_ladders.json` | pooled statistics over those ladders | `build/llm_ladders.py` |
| `final_numbers.json` | implied valuations for the deployment-cost test | `scripts/external_criterion.py` |
| `endpoints.json` | conjugate endpoint checks | `scripts/verify_endpoints.py` |
| `minimax_verification.json` | Bayes and minimax grid searches | `scripts/verify_minimax.py` |

`build/llm_ladders.py` additionally needs `huggingface_v1.csv` and
`huggingface_v2.csv`, snapshots of the Open LLM Leaderboard. Regenerate them
with `build/fetch_llm_sources.py`; note that the live leaderboard moves, so an
exact re-fetch requires the contemporaneous revision.

### A known limitation

The `timm` accuracy and latency tables are append-only and historically stable.
The Ultralytics documentation is living text, and several detection entries have
migrated between device columns since extraction. Exact reproduction of the
three Ultralytics cells therefore needs the documentation at the contemporaneous
commit. This affects none of the conclusions: the nine `timm` cells alone give a
median price of 0.344 against 0.3439 for all twelve.

---

## Pipeline stages

`run_all.sh` executes these in order. Each is independently runnable with
`DATA=data OUT=results python3 scripts/<name>.py`.

**Primary corpus.** `analysis_main.py` fits every frontier, checks the duality
and regret identities and runs the decision experiments. `stats_meta.py` adds
the bootstrap, random-effects pooling, AICc model selection, split-half
decircularisation and the numerical Bayes/minimax verification.
`floor_identification.py` applies the identification rule.
`decision_logspace.py` repeats the decision experiment in log space.

**Transfer and scope.** `stats_extended.py` (group O), `stats_llm.py`,
`latency_ladders.py`, `latency_final.py` and `joint_llm.py` (group L),
`analysis_retired.py` and `retired_stats.py` (group R, including the censoring
analysis).

**Audits.** `bias_audit.py` (Pareto-selection and winner's-curse bias),
`noise_model.py` (test-set noise), `verify_minimax.py` and
`verify_endpoints.py` (numerical checks of the theorems), `reconcile.py` and
`finalize.py` (cross-checks that every quoted scalar agrees across stages).

**External criterion.** `external_criterion.py` implements the deployment-cost
verdict of Section 6. Run `--selftest` to verify the Proposition 22 identities
symbolically-in-effect; run with `--indir`/`--outdir` to score it on data.

**Robustness (slow).** `robustness.py` runs the winner's-curse simulation,
per-cell parametric calibration, system-level bootstrap, clustered pooling,
identifiability sensitivity and metric dependence. It writes a checkpoint after
each stage, so an interrupted run leaves a partial but valid
`robustness.json` carrying a `_stages_completed` list and a `_complete` flag.

**Figures.** `make_figs_is.py` (Figures 1, 4, 6, 7, 8), `make_figs_ext.py`
(9), `make_fig_llm.py` (10), `make_fig_ladders.py` (11),
`make_fig_retired.py` (12), `make_figure.py` (the economics figure).

---

## The selection rule, in ten lines

The whole method, for a practitioner who wants nothing else from this archive:

```python
import math

def select(systems, q0, lam=0.35):
    """systems: list of (name, quality Q, resource t>0); q0: chance level.
    Returns the selected name and the margin over the runner-up, in nats."""
    scored = []
    for name, Q, t in systems:
        eps = 1.0 - (Q - q0) / (1.0 - q0)        # chance-corrected error
        scored.append((-math.log(eps) - lam * math.log(t), name))
    scored.sort(reverse=True)
    margin = scored[0][0] - scored[1][0] if len(scored) > 1 else float('inf')
    return scored[0][1], margin
```

A margin below about 0.05 nats means the systems are close substitutes and the
verdict should not be leaned on. Quote the resource and the quality metric
alongside any price: a price is a property of a (frontier, resource, metric)
triple, and 0.35 is the value for inference latency measured on a
chance-corrected error scale.

---

## Licence

Code is MIT (`LICENSE`). Data compilations are CC BY 4.0 (`LICENSE-DATA`).
The underlying upstream benchmarks remain under their own licences; this archive
redistributes measurements drawn from public leaderboards and documentation, with
sources given above.

## Contact

Dmitry Ignatov — Computer Vision Lab, CAIDAS & IFI, University of Würzburg
