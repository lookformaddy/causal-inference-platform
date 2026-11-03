"""Sensitivity analysis: simplified Rosenbaum gamma bounds.

Question answered: how strong would an unobserved confounder have to be
to explain away the observed effect? For matched pairs, let Gamma be the
maximum odds ratio by which hidden bias could tilt treatment assignment
within a pair. Under Gamma, the probability that the treated unit in a
pair is the one with the higher outcome is bounded by p+ = Gamma/(1+Gamma).

For each Gamma we compute the *upper-bound* p-value of a sign test
(counting pairs where the treated outcome exceeds the control outcome)
under a Binomial(n_pairs, p+) null. The reported "critical Gamma" is the
smallest Gamma at which the effect is no longer significant at alpha —
a larger critical Gamma means a more robust finding. Gamma = 1 is the
no-hidden-bias case.

This is a simplified bound (sign test rather than the full Wilcoxon
signed-rank bound); the simplification is documented, not hidden, and
the function is explicit about which pairs it uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed in log space."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    # Sum pmf from k..n; n is pair counts (hundreds), direct sum is fine.
    for j in range(k, n + 1):
        log_pmf = (
            math.lgamma(n + 1)
            - math.lgamma(j + 1)
            - math.lgamma(n - j + 1)
            + j * math.log(p)
            + (n - j) * math.log(1.0 - p)
        )
        total += math.exp(log_pmf)
    return min(total, 1.0)


@dataclass
class SensitivityResult:
    n_pairs: int
    n_positive: int  # pairs where treated outcome > control outcome
    gammas: list[float]
    upper_p_values: list[float]
    critical_gamma: float | None  # first gamma where p > alpha, else None
    alpha: float = 0.05

    def as_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "n_positive": self.n_positive,
            "bounds": [
                {"gamma": g, "upper_p_value": p}
                for g, p in zip(self.gammas, self.upper_p_values)
            ],
            "critical_gamma": self.critical_gamma,
            "alpha": self.alpha,
        }


def rosenbaum_bounds(
    outcome: list[float],
    pairs: list[tuple[int, int]],
    gammas: list[float] | None = None,
    alpha: float = 0.05,
) -> SensitivityResult:
    """Rosenbaum-style gamma bounds for matched-pair sign-test results."""
    if not pairs:
        raise ValueError("need matched pairs for sensitivity analysis")
    gammas = gammas or [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    diffs = [outcome[t] - outcome[c] for t, c in pairs]
    # Ties carry no sign information and are dropped (standard practice).
    signs = [d for d in diffs if d != 0]
    n = len(signs)
    n_positive = sum(1 for d in signs if d > 0)
    upper_p: list[float] = []
    critical: float | None = None
    for gamma in gammas:
        p_plus = gamma / (1.0 + gamma)
        p_val = _binom_sf(n_positive, n, p_plus)
        upper_p.append(p_val)
        if critical is None and p_val > alpha:
            critical = gamma
    return SensitivityResult(
        n_pairs=n,
        n_positive=n_positive,
        gammas=list(gammas),
        upper_p_values=upper_p,
        critical_gamma=critical,
        alpha=alpha,
    )
