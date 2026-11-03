"""Propensity model, matching, balance diagnostics, and recovery of a
known effect from confounded synthetic data."""

from causal_inference import (
    fit_propensity,
    generate_observational,
    matched_ate,
    nearest_neighbour_match,
)
from causal_inference.matching import balance_report


def test_propensity_scores_are_probabilities():
    data = generate_observational(n=500, seed=1)
    _, scores = fit_propensity(data.covariates, data.treatment)
    assert len(scores) == 500
    assert all(0.0 < p < 1.0 for p in scores)
    # Treated units should have higher scores on average (confounded data).
    treated_mean = sum(s for s, t in zip(scores, data.treatment) if t == 1) / sum(data.treatment)
    control_mean = sum(s for s, t in zip(scores, data.treatment) if t == 0) / (500 - sum(data.treatment))
    assert treated_mean > control_mean


def test_matching_improves_balance():
    data = generate_observational(n=2000, seed=42)
    model, _ = fit_propensity(data.covariates, data.treatment)
    match = nearest_neighbour_match(model.logit(data.covariates), data.treatment)
    report = balance_report(
        data.covariates, data.covariate_names, data.treatment, match.pairs
    )
    assert len(match.pairs) > 500  # most treated units matched
    assert report.max_abs_smd_after < report.max_abs_smd_before
    assert report.balanced


def test_matched_ate_recovers_true_effect():
    data = generate_observational(n=2000, true_effect=0.8, seed=42)
    model, _ = fit_propensity(data.covariates, data.treatment)
    match = nearest_neighbour_match(model.logit(data.covariates), data.treatment)
    assert abs(matched_ate(data.outcome, match.pairs) - 0.8) < 0.25


def test_naive_difference_is_biased():
    """Sanity: the generator really is confounded, so adjustment matters."""
    data = generate_observational(n=2000, true_effect=0.8, seed=42)
    naive = sum(data.outcome[i] for i in data.treated()) / len(data.treated())
    naive -= sum(data.outcome[i] for i in data.control()) / len(data.control())
    assert naive - 0.8 > 0.3  # naive overstates the effect


def test_unmatched_treated_reported_not_forced():
    data = generate_observational(n=200, seed=3)
    model, _ = fit_propensity(data.covariates, data.treatment)
    # Absurdly tight caliper forces unmatched units.
    match = nearest_neighbour_match(
        model.logit(data.covariates), data.treatment, caliper=0.001
    )
    assert match.unmatched_treated
    assert len(match.pairs) + len(match.unmatched_treated) == sum(data.treatment)
