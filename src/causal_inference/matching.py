"""Nearest-neighbour propensity matching with balance diagnostics.

Each treated unit is matched to the closest control unit on the logit
propensity score, subject to a caliper (default 0.2 * SD of the logit,
the standard Rosenbaum & Rubin recommendation). Matching is without
replacement in a deterministic order: treated units are processed in
order of decreasing match difficulty (largest minimum distance first),
which avoids the greedy-order pathology of first-come matching.

Balance is judged by the standardized mean difference (SMD) per
covariate; |SMD| < 0.1 is the conventional "balanced" threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pooled_std(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 1.0
    ma, mb = _mean(a), _mean(b)
    va = sum((v - ma) ** 2 for v in a) / (len(a) - 1)
    vb = sum((v - mb) ** 2 for v in b) / (len(b) - 1)
    pooled = math.sqrt((va + vb) / 2.0)
    return pooled or 1.0


def standardized_mean_difference(
    treated_values: list[float], control_values: list[float]
) -> float:
    """SMD = (mean_t - mean_c) / pooled SD. Sign preserved."""
    return (_mean(treated_values) - _mean(control_values)) / _pooled_std(
        treated_values, control_values
    )


@dataclass
class BalanceReport:
    """Per-covariate SMD before and after matching."""

    covariate_names: list[str]
    smd_before: dict[str, float]
    smd_after: dict[str, float]
    n_matched_pairs: int

    @property
    def max_abs_smd_before(self) -> float:
        return max((abs(v) for v in self.smd_before.values()), default=0.0)

    @property
    def max_abs_smd_after(self) -> float:
        return max((abs(v) for v in self.smd_after.values()), default=0.0)

    @property
    def balanced(self) -> bool:
        """Conventional gate: every covariate |SMD| < 0.1 after matching."""
        return self.max_abs_smd_after < 0.1

    def as_dict(self) -> dict:
        return {
            "covariate_names": self.covariate_names,
            "smd_before": self.smd_before,
            "smd_after": self.smd_after,
            "max_abs_smd_before": self.max_abs_smd_before,
            "max_abs_smd_after": self.max_abs_smd_after,
            "balanced": self.balanced,
            "n_matched_pairs": self.n_matched_pairs,
        }


@dataclass
class MatchResult:
    pairs: list[tuple[int, int]]  # (treated index, control index)
    unmatched_treated: list[int] = field(default_factory=list)


def nearest_neighbour_match(
    logit_scores: list[float],
    treatment: list[int],
    caliper: float | None = None,
) -> MatchResult:
    """Greedy 1:1 nearest-neighbour matching on logit propensity.

    If ``caliper`` is None it defaults to 0.2 * SD(logit). A treated unit
    with no control inside the caliper is left unmatched and reported,
    never force-matched — a silent source of bias in naive implementations.
    """
    treated_idx = [i for i, t in enumerate(treatment) if t == 1]
    control_idx = [i for i, t in enumerate(treatment) if t == 0]
    if caliper is None:
        mean = _mean(logit_scores)
        var = sum((v - mean) ** 2 for v in logit_scores) / max(len(logit_scores) - 1, 1)
        caliper = 0.2 * math.sqrt(var)

    # Hardest-to-match first: sort treated units by their closest-control distance.
    def closest_distance(t: int) -> float:
        return min(abs(logit_scores[t] - logit_scores[c]) for c in control_idx)

    available = set(control_idx)
    pairs: list[tuple[int, int]] = []
    unmatched: list[int] = []
    for t in sorted(treated_idx, key=closest_distance, reverse=True):
        candidates = [c for c in available if abs(logit_scores[t] - logit_scores[c]) <= caliper]
        if not candidates:
            unmatched.append(t)
            continue
        best = min(candidates, key=lambda c: abs(logit_scores[t] - logit_scores[c]))
        pairs.append((t, best))
        available.discard(best)
    return MatchResult(pairs=pairs, unmatched_treated=unmatched)


def matched_ate(
    outcome: list[float], pairs: list[tuple[int, int]]
) -> float:
    """Average treatment effect on the treated, from matched pairs."""
    if not pairs:
        raise ValueError("no matched pairs: cannot estimate an effect")
    return _mean([outcome[t] - outcome[c] for t, c in pairs])


def balance_report(
    covariates: list[list[float]],
    covariate_names: list[str],
    treatment: list[int],
    pairs: list[tuple[int, int]],
) -> BalanceReport:
    """SMD per covariate before matching and on the matched sample."""
    treated_idx = [i for i, t in enumerate(treatment) if t == 1]
    control_idx = [i for i, t in enumerate(treatment) if t == 0]
    matched_t = [t for t, _ in pairs]
    matched_c = [c for _, c in pairs]
    before: dict[str, float] = {}
    after: dict[str, float] = {}
    for j, name in enumerate(covariate_names):
        col = [row[j] for row in covariates]
        before[name] = standardized_mean_difference(
            [col[i] for i in treated_idx], [col[i] for i in control_idx]
        )
        after[name] = standardized_mean_difference(
            [col[i] for i in matched_t], [col[i] for i in matched_c]
        )
    return BalanceReport(
        covariate_names=covariate_names,
        smd_before=before,
        smd_after=after,
        n_matched_pairs=len(pairs),
    )
