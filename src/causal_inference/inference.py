"""Bootstrap confidence intervals for any row-resampling estimator.

The estimator is passed as a callable ``estimate(rows)`` where ``rows``
is a resampled index list — this keeps the bootstrap generic over
matching, IPW, and DiD estimators instead of re-implementing resampling
per method. Percentile intervals; the seed makes the interval
reproducible, which matters when a CI gates a promotion decision.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass
class ConfidenceInterval:
    point: float
    lower: float
    upper: float
    n_bootstraps: int
    alpha: float = 0.05

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0 or self.upper < 0

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width,
            "excludes_zero": self.excludes_zero,
            "n_bootstraps": self.n_bootstraps,
            "alpha": self.alpha,
        }


def bootstrap_ci(
    estimate_fn: Callable[[list[int]], float],
    n: int,
    point: float,
    n_bootstraps: int = 200,
    alpha: float = 0.05,
    seed: int = 123,
) -> ConfidenceInterval:
    """Percentile bootstrap CI for ``estimate_fn`` over ``n`` rows."""
    if n < 2:
        raise ValueError("need at least 2 rows to bootstrap")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_bootstraps):
        sample = [rng.randrange(n) for _ in range(n)]
        try:
            estimates.append(estimate_fn(sample))
        except ValueError:
            # e.g. a resample with no matched pairs: skip, don't crash.
            continue
    if len(estimates) < 10:
        raise ValueError("too few successful bootstrap replicates")
    estimates.sort()
    lo_idx = int((alpha / 2) * len(estimates))
    hi_idx = min(int((1 - alpha / 2) * len(estimates)), len(estimates) - 1)
    return ConfidenceInterval(
        point=point,
        lower=estimates[lo_idx],
        upper=estimates[hi_idx],
        n_bootstraps=len(estimates),
        alpha=alpha,
    )
