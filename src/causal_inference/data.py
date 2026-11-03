"""Deterministic synthetic data generators.

Every dataset is synthetic, generated from a seeded PRNG, with a *known*
true effect. That is what makes the demo verifiable: an estimator that
cannot recover the planted effect on data it was designed for is broken.

Domains are deliberately generic (a "program" offered to "units") so no
real-world employer, product, or customer data is involved.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class SyntheticData:
    """One cross-sectional observational dataset."""

    covariates: list[list[float]]  # n x p, column names in `covariate_names`
    covariate_names: list[str]
    treatment: list[int]  # 0 / 1
    outcome: list[float]
    true_effect: float
    n: int = 0

    def __post_init__(self) -> None:
        self.n = len(self.treatment)

    def treated(self) -> list[int]:
        return [i for i, t in enumerate(self.treatment) if t == 1]

    def control(self) -> list[int]:
        return [i for i, t in enumerate(self.treatment) if t == 0]


def generate_observational(
    n: int = 2000,
    true_effect: float = 0.8,
    seed: int = 42,
    confounding: float = 1.0,
) -> SyntheticData:
    """Confounded observational data.

    Covariates drive both treatment assignment and the outcome, so the
    naive treated-minus-control difference is biased upward: units with
    high x1 are both more likely to be treated and have higher outcomes
    regardless of treatment. Set ``confounding=0`` for a randomized
    benchmark where the naive difference is unbiased.
    """
    rng = random.Random(seed)
    names = ["x1", "x2", "x3"]
    covariates: list[list[float]] = []
    treatment: list[int] = []
    outcome: list[float] = []
    for _ in range(n):
        x1 = rng.gauss(0.0, 1.0)
        x2 = rng.gauss(0.0, 1.0)
        x3 = rng.gauss(0.0, 1.0)
        # Assignment: logistic in the covariates (confounded when > 0).
        logit = confounding * (-0.5 + 0.8 * x1 + 0.5 * x2 - 0.4 * x3)
        treated = 1 if rng.random() < _sigmoid(logit) else 0
        y = 2.0 + 1.0 * x1 + 0.5 * x2 + 0.3 * x3 + true_effect * treated
        y += rng.gauss(0.0, 1.0)
        covariates.append([x1, x2, x3])
        treatment.append(treated)
        outcome.append(y)
    return SyntheticData(
        covariates=covariates,
        covariate_names=names,
        treatment=treatment,
        outcome=outcome,
        true_effect=true_effect,
    )


@dataclass
class PanelData:
    """Two-group, multi-period panel for difference-in-differences."""

    unit_id: list[int]
    period: list[int]  # -2, -1 pre-periods; 0 post-period
    treated_group: list[int]  # 1 if the unit's group ever receives treatment
    outcome: list[float]
    true_effect: float
    post_period: int = 0
    pre_periods: list[int] = field(default_factory=lambda: [-2, -1])


def generate_panel(
    n_units: int = 800,
    true_effect: float = 0.6,
    seed: int = 7,
) -> PanelData:
    """Panel with a common time trend and a group fixed effect.

    Both groups share the same pre-period trend (parallel trends holds by
    construction), and only the treated group jumps by ``true_effect`` in
    the post period. A placebo DiD between the two pre-periods should
    therefore be near zero.
    """
    rng = random.Random(seed)
    unit_id: list[int] = []
    period: list[int] = []
    treated_group: list[int] = []
    outcome: list[float] = []
    for u in range(n_units):
        group = 1 if rng.random() < 0.5 else 0
        base = rng.gauss(0.0, 1.0)
        group_effect = 0.4 * group  # level shift, not a trend
        for p in (-2, -1, 0):
            trend = 0.2 * p  # common to both groups
            post_effect = true_effect * group if p == 0 else 0.0
            y = 3.0 + base + group_effect + trend + post_effect + rng.gauss(0.0, 0.5)
            unit_id.append(u)
            period.append(p)
            treated_group.append(group)
            outcome.append(y)
    return PanelData(
        unit_id=unit_id,
        period=period,
        treated_group=treated_group,
        outcome=outcome,
        true_effect=true_effect,
    )
