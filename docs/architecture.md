# Architecture

## Overview

```
                    ┌──────────────────────┐
                    │  Synthetic data      │
                    │  (known true effect) │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
   ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐
   │ Propensity model│ │ Panel data   │ │ Outcome models│
   │ (logistic, from │ │ (treated /   │ │ (OLS per arm, │
   │  scratch)       │ │  control x   │ │  T-learner)   │
   └────────┬────────┘ │  pre / post) │ └───────┬───────┘
            │          └──────┬───────┘         │
            ▼                 ▼                 ▼
   ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐
   │ Matching (1:1,  │ │ DiD + placebo│ │ Uplift deciles│
   │ caliper) + SMD  │ │ parallel-    │ │ (ranked by    │
   │ diagnostics     │ │ trends check │ │  predicted    │
   └────────┬────────┘ └──────┬───────┘ │  uplift)      │
            │                 │         └───────┬───────┘
            ▼                 ▼                 ▼
   ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐
   │ IPW (stabilized │ │ Bootstrap CI │ │ Rosenbaum     │
   │ Hajek, clipped) │ │ (percentile) │ │ gamma bounds  │
   └────────┬────────┘ └──────┬───────┘ └───────┬───────┘
            │                 │                 │
            └─────────────────┴─────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ Experiment registry           │
              │ (versioned, append-only,      │
              │  fail-closed promotion gates) │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ Optional FastAPI service      │
              │ (POST /analyze, GET /health)  │
              └───────────────────────────────┘
```

## Module contracts

| Module | Input | Output | Key invariant |
|---|---|---|---|
| `propensity` | covariates, treatment | model + scores in (0, 1) | Features standardized internally; scoring reuses stored parameters |
| `matching` | logit scores, treatment | pairs + unmatched list | No treated unit is force-matched outside the caliper |
| `weighting` | outcome, treatment, propensity | Hajek IPW estimate + diagnostics | Scores clipped to [0.01, 0.99]; clipping count reported |
| `did` | panel data | DiD estimate + placebo + flag | Placebo uses only pre-periods, so it cannot leak the post effect |
| `uplift` | covariates, treatment, outcome | per-decile observed uplift | Deciles partition all units exactly once |
| `inference` | estimator callable, n, point | percentile CI | Failed replicates (e.g. empty resample) are skipped, not fatal |
| `sensitivity` | outcome, matched pairs | gamma → upper p-value table | p-values non-decreasing in gamma |
| `registry` | experiment record | PROMOTED or REJECTED + reasons | Versions append-only; promotion requires every gate |

## Matching algorithm

1. Compute logit propensity scores.
2. Caliper = 0.2 × SD(logit) unless overridden.
3. Sort treated units by distance to their nearest available control,
   hardest first.
4. Greedily assign the nearest control within the caliper; remove it
   from the pool. Units with no control inside the caliper go to
   `unmatched_treated` and are reported in every summary.

Complexity is O(T × C) in the worst case — fine for analysis-sized data;
a production deployment would index controls (e.g. KD-tree) before
scaling past ~100k units.

## Promotion gates

Default gate (`PromotionGate`): max |SMD| after matching < 0.1,
sample ≥ 100, CI excludes zero, parallel trends holds (when the
experiment carries a DiD check), and — when configured — critical Gamma
above a floor. Every failed gate contributes a human-readable reason;
the registry stores the reasons on the experiment record.

## Data flow and determinism

All generators take an explicit seed. The demo uses seed 42
(observational, true effect 0.8) and seed 7 (panel, true effect 0.6),
so `python demo.py` is reproducible and its assertions check the
adjusted estimates against ground truth rather than printing numbers
nobody verifies.

## Diagram description (for rendering)

A left-to-right pipeline: synthetic data feeds three estimation paths
(propensity → matching/IPW, panel → DiD, outcome models → uplift).
Uncertainty (bootstrap CI) and robustness (Rosenbaum bounds) wrap all
three paths. Everything converges on the versioned experiment registry,
whose fail-closed gates decide promotion. An optional FastAPI layer
exposes the observational path as `POST /analyze`.
