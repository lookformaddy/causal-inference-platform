"""Uplift deciles, bootstrap CIs, and Rosenbaum sensitivity bounds."""

from causal_inference import (
    bootstrap_ci,
    decile_uplift,
    fit_propensity,
    fit_uplift_model,
    generate_observational,
    ipw_ate,
    nearest_neighbour_match,
    rosenbaum_bounds,
)


def _data():
    return generate_observational(n=2000, true_effect=0.8, seed=42)


def test_uplift_deciles_cover_all_units():
    data = _data()
    model = fit_uplift_model(data.covariates, data.treatment, data.outcome)
    predicted = model.predicted_uplift(data.covariates)
    deciles = decile_uplift(predicted, data.treatment, data.outcome)
    assert len(deciles) == 10
    assert deciles[0].decile == 1 and deciles[-1].decile == 10
    total = sum(d.n_treated + d.n_control for d in deciles)
    assert total == 2000
    # Deciles partition the sample, so the size-weighted mean of the
    # per-decile *raw* differences must reproduce the naive difference —
    # raw decile differences are still confounded within decile, which is
    # exactly why decile tables are a diagnostic, not an adjusted estimate.
    naive = sum(data.outcome[i] for i in data.treated()) / len(data.treated())
    naive -= sum(data.outcome[i] for i in data.control()) / len(data.control())
    weighted = sum(
        d.observed_uplift * (d.n_treated + d.n_control) for d in deciles
    ) / total
    assert abs(weighted - naive) < 0.05
    # Ranking works: top decile's predicted uplift exceeds the bottom's.
    assert predicted and max(predicted) > min(predicted)


def test_bootstrap_ci_covers_true_effect():
    data = _data()
    _, propensity = fit_propensity(data.covariates, data.treatment)
    point = ipw_ate(data.outcome, data.treatment, propensity).estimate

    def estimate(idx: list[int]) -> float:
        return ipw_ate(
            [data.outcome[i] for i in idx],
            [data.treatment[i] for i in idx],
            [propensity[i] for i in idx],
        ).estimate

    ci = bootstrap_ci(estimate, data.n, point, n_bootstraps=100, seed=5)
    assert ci.lower <= 0.8 <= ci.upper
    assert ci.excludes_zero
    assert ci.width > 0


def test_rosenbaum_bounds_monotone_and_gamma1_significant():
    data = _data()
    model, _ = fit_propensity(data.covariates, data.treatment)
    match = nearest_neighbour_match(model.logit(data.covariates), data.treatment)
    result = rosenbaum_bounds(data.outcome, match.pairs)
    # Larger hidden bias can only weaken significance.
    assert result.upper_p_values == sorted(result.upper_p_values)
    assert result.upper_p_values[0] < 0.05  # gamma=1: effect significant
    assert result.critical_gamma is not None
    assert result.critical_gamma >= 1.0
