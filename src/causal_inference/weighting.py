"""Inverse-propensity weighting (IPW).

Two estimators, both using stabilized weights to tame variance:
Hajek (ratio) IPW for the ATE, and the Horvitz-Thompson form. Scores
are clipped to [eps, 1-eps] (default eps=0.01) — unclipped weights from
near-deterministic assignment explode, and clipping is reported in the
diagnostics rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IPWResult:
    estimate: float
    n: int
    n_treated: int
    n_control: int
    max_weight: float
    mean_weight: float
    n_clipped: int


def ipw_ate(
    outcome: list[float],
    treatment: list[int],
    propensity: list[float],
    clip: float = 0.01,
    stabilized: bool = True,
) -> IPWResult:
    """Hajek IPW estimate of the average treatment effect."""
    if not (len(outcome) == len(treatment) == len(propensity)):
        raise ValueError("outcome, treatment and propensity must align")
    clipped = [min(max(p, clip), 1.0 - clip) for p in propensity]
    n_clipped = sum(1 for p, c in zip(propensity, clipped) if p != c)
    n = len(outcome)
    p_treat = sum(treatment) / n if stabilized else 1.0
    weights: list[float] = []
    weighted_outcome_sum = 0.0
    for y, t, p in zip(outcome, treatment, clipped):
        w = (p_treat / p) if t == 1 else ((1.0 - p_treat) / (1.0 - p))
        weights.append(w)
        weighted_outcome_sum += w * (y if t == 1 else -y)
    # Hajek form: normalize by the sum of weights in each arm.
    treated_weight = sum(w for w, t in zip(weights, treatment) if t == 1)
    control_weight = sum(w for w, t in zip(weights, treatment) if t == 0)
    mu1 = sum(w * y for w, y, t in zip(weights, outcome, treatment) if t == 1) / treated_weight
    mu0 = sum(w * y for w, y, t in zip(weights, outcome, treatment) if t == 0) / control_weight
    return IPWResult(
        estimate=mu1 - mu0,
        n=n,
        n_treated=sum(treatment),
        n_control=n - sum(treatment),
        max_weight=max(weights),
        mean_weight=sum(weights) / n,
        n_clipped=n_clipped,
    )
