"""Difference-in-differences (DiD) with a parallel-trends check.

The DiD estimate is (treated post - treated pre) - (control post -
control pre), using group means. The parallel-trends check is a placebo
DiD computed between two pre-periods: if the groups were already
diverging before treatment, the post-period DiD is not credible and the
check fails. A DiD that ships without this check is an opinion, not an
estimate — so the check is a first-class return value, not an appendix.
"""

from __future__ import annotations

from dataclasses import dataclass

from causal_inference.data import PanelData


def _group_mean(data: PanelData, group: int, periods: list[int]) -> float:
    values = [
        y
        for y, g, p in zip(data.outcome, data.treated_group, data.period)
        if g == group and p in periods
    ]
    if not values:
        raise ValueError(f"no observations for group={group}, periods={periods}")
    return sum(values) / len(values)


@dataclass
class DiDResult:
    estimate: float
    treated_pre: float
    treated_post: float
    control_pre: float
    control_post: float
    placebo_estimate: float
    parallel_trends_ok: bool
    placebo_threshold: float = 0.15

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "treated_pre": self.treated_pre,
            "treated_post": self.treated_post,
            "control_pre": self.control_pre,
            "control_post": self.control_post,
            "placebo_estimate": self.placebo_estimate,
            "parallel_trends_ok": self.parallel_trends_ok,
        }


def did_estimate(data: PanelData, placebo_threshold: float = 0.15) -> DiDResult:
    """DiD on the post period vs all pre-periods, plus a placebo check.

    The placebo DiD compares the *last* pre-period against the *earlier*
    pre-periods (both untreated for both groups). If its magnitude
    exceeds ``placebo_threshold`` the parallel-trends assumption is
    flagged as violated.
    """
    treated_pre = _group_mean(data, 1, data.pre_periods)
    treated_post = _group_mean(data, 1, [data.post_period])
    control_pre = _group_mean(data, 0, data.pre_periods)
    control_post = _group_mean(data, 0, [data.post_period])
    estimate = (treated_post - treated_pre) - (control_post - control_pre)

    placebo = 0.0
    if len(data.pre_periods) >= 2:
        last_pre = [data.pre_periods[-1]]
        earlier_pre = data.pre_periods[:-1]
        placebo = (
            _group_mean(data, 1, last_pre) - _group_mean(data, 1, earlier_pre)
        ) - (
            _group_mean(data, 0, last_pre) - _group_mean(data, 0, earlier_pre)
        )
    return DiDResult(
        estimate=estimate,
        treated_pre=treated_pre,
        treated_post=treated_post,
        control_pre=control_pre,
        control_post=control_post,
        placebo_estimate=placebo,
        parallel_trends_ok=abs(placebo) <= placebo_threshold,
        placebo_threshold=placebo_threshold,
    )


def parallel_trends_check(data: PanelData, placebo_threshold: float = 0.15) -> bool:
    """Convenience wrapper: True when the placebo DiD is small."""
    return did_estimate(data, placebo_threshold).parallel_trends_ok
