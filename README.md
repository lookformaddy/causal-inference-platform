# Causal Inference Platform

An enterprise-grade causal inference platform, implemented from scratch in
pure Python standard library — no sklearn, no statsmodels, no scipy. It
answers the question every experimentation and measurement team eventually
faces: **did the program cause the outcome, or were the treated units
already different?**

## The enterprise problem

Observational data is confounded by default: the units that receive a
program, feature, or intervention are rarely a random sample. A naive
treated-minus-control comparison credits the program with pre-existing
differences. Shipping decisions made on naive comparisons waste budget
on programs that never worked and kill programs that did. This platform
provides the full measurement pipeline, with every assumption checked
explicitly rather than assumed silently:

1. **Propensity modeling** — from-scratch logistic regression estimates
   each unit's probability of treatment given its covariates.
2. **Matching** — 1:1 nearest-neighbour matching on the logit propensity
   score with a caliper; units that cannot be matched are reported, never
   force-matched.
3. **Balance diagnostics** — standardized mean differences (SMD) per
   covariate before and after matching; |SMD| < 0.1 is the balance gate.
4. **Inverse-propensity weighting (IPW)** — a second, independent
   estimator of the same effect, with stabilized weights and clipping
   diagnostics, so two methods must agree before anyone trusts a number.
5. **Difference-in-differences (DiD)** — for panel data, with a
   placebo DiD between pre-periods as a first-class parallel-trends check.
6. **Uplift / decile analysis** — a two-model (T-learner) uplift model
   ranks units by predicted uplift; per-decile observed differences show
   where the effect concentrates and whether targeting beats treating
   everyone.
7. **Bootstrap confidence intervals** — percentile intervals over any
   row-resampling estimator, seeded for reproducibility.
8. **Sensitivity analysis** — simplified Rosenbaum gamma bounds: how
   strong would an unobserved confounder have to be to explain the effect
   away? The critical Gamma is reported with every result.
9. **Experiment registry with promotion gates** — versioned, append-only
   experiment records. Promotion is fail-closed: balance, CI excluding
   zero, minimum sample, parallel trends, and a sensitivity floor must
   all pass, or the experiment is rejected with reasons.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full design,
including the estimator contracts, the matching algorithm, and the gate
semantics.

```
src/causal_inference/
  data.py        deterministic synthetic generators (known true effects)
  propensity.py  logistic regression from scratch (gradient descent)
  matching.py    caliper nearest-neighbour matching + SMD diagnostics
  weighting.py   stabilized Hajek IPW with clipping diagnostics
  did.py         difference-in-differences + placebo parallel-trends check
  uplift.py      T-learner uplift model + decile analysis (OLS from scratch)
  inference.py   generic percentile bootstrap CI
  sensitivity.py Rosenbaum-style gamma bounds (sign test)
  registry.py    versioned experiment registry + promotion gates
  service.py     optional FastAPI service (skipped without FastAPI)
tests/           pytest suite; service tests auto-skip without FastAPI
demo.py          end-to-end demo recovering planted effects
```

## How to run it

```bash
pip install -r requirements.txt   # pytest only; core is stdlib-only
pytest -q                         # full test suite
python demo.py                    # end-to-end demo
make test && make demo
```

Optional API (requires FastAPI + uvicorn, see `requirements.txt`):

```bash
pip install fastapi uvicorn
uvicorn causal_inference.service:app --reload  # PYTHONPATH=src
curl -X POST localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"n": 2000, "true_effect": 0.8, "seed": 42}'
```

Docker:

```bash
docker build -t causal-inference-platform .
docker run --rm causal-inference-platform
```

## Key design decisions and trade-offs

- **Pure stdlib core.** No sklearn/statsmodels means the platform runs
  anywhere, has no version-pinning risk, and every estimator's math is
  auditable in one file. The cost is speed on very large data and the
  absence of battle-tested solvers — acceptable for a measurement
  platform whose datasets are analysis-sized, not training-sized.
- **Matching without replacement, hardest-first.** Treated units are
  processed in order of decreasing match difficulty, avoiding the
  greedy-order pathology where early easy matches steal controls that
  hard units needed. Unmatched units are reported, not silently dropped
  from the denominator.
- **Caliper default 0.2 × SD(logit).** The Rosenbaum & Rubin standard.
  A tighter caliper improves balance but discards treated units and
  changes the estimand (ATT on the matched subset, not all treated) —
  the demo prints the unmatched count so that trade-off stays visible.
- **Two estimators, one decision.** Matching and IPW use the same
  propensity scores but different identifying machinery; agreement
  between them is a robustness signal, disagreement is a reason to stop.
- **Bootstrap holds propensity scores fixed.** Refitting the propensity
  model per replicate is more honest but ~100× slower; the API accepts
  any row-resampling estimator, so a caller can pay that cost when it
  matters.
- **Simplified Rosenbaum bounds.** The sign-test bound is used instead
  of the full Wilcoxon signed-rank bound: easier to audit, slightly
  conservative, and documented as simplified in the module docstring.
- **Fail-closed registry.** Promotion requires every gate to pass; a
  rejected experiment keeps its reasons attached, so a decision is
  always traceable to the numbers that justified or blocked it.

## What it demonstrates

- Causal inference end-to-end: confounding, propensity modeling,
  matching, weighting, panel DiD, heterogeneity, uncertainty, and
  sensitivity — not just one estimator in isolation.
- From-scratch statistical machinery (logistic regression, OLS,
  bootstrap, binomial bounds) implemented and tested without a stats
  library.
- Measurement governance: balance gates, placebo checks, sensitivity
  floors, and a versioned promotion workflow — the difference between a
  notebook estimate and a decision a business can defend.
- Clean-room discipline: all data is synthetic with planted effects, so
  every estimate in the demo is verifiable against ground truth.
```

## Limitations (stated, not hidden)

- Propensity methods assume unconfoundedness given observed covariates;
  the sensitivity analysis quantifies — but cannot remove — hidden bias.
- DiD assumes parallel trends; the placebo check tests it on pre-periods
  but cannot prove it for the post period.
- The T-learner uplift model is a diagnostic for heterogeneity, not a
  causal claim about any single decile.
