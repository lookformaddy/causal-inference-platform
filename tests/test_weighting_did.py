"""IPW estimation and difference-in-differences."""

from causal_inference import (
    did_estimate,
    fit_propensity,
    generate_observational,
    generate_panel,
    ipw_ate,
)


def test_ipw_recovers_true_effect():
    data = generate_observational(n=2000, true_effect=0.8, seed=42)
    _, propensity = fit_propensity(data.covariates, data.treatment)
    result = ipw_ate(data.outcome, data.treatment, propensity)
    assert abs(result.estimate - 0.8) < 0.25
    assert result.max_weight < 200  # clipping keeps weights sane
    assert result.n_treated + result.n_control == 2000


def test_ipw_on_randomized_data_matches_naive():
    data = generate_observational(n=2000, true_effect=0.8, seed=9, confounding=0.0)
    _, propensity = fit_propensity(data.covariates, data.treatment)
    ipw = ipw_ate(data.outcome, data.treatment, propensity).estimate
    naive = sum(data.outcome[i] for i in data.treated()) / len(data.treated())
    naive -= sum(data.outcome[i] for i in data.control()) / len(data.control())
    assert abs(ipw - naive) < 0.3


def test_did_recovers_true_effect_and_parallel_trends():
    panel = generate_panel(n_units=800, true_effect=0.6, seed=7)
    result = did_estimate(panel)
    assert abs(result.estimate - 0.6) < 0.25
    assert result.parallel_trends_ok
    assert abs(result.placebo_estimate) <= 0.15


def test_did_flags_broken_parallel_trends():
    # Inject a pre-period divergence: treated group trends up before treatment.
    panel = generate_panel(n_units=800, true_effect=0.6, seed=7)
    tampered = list(panel.outcome)
    for i, (g, p) in enumerate(zip(panel.treated_group, panel.period)):
        if g == 1 and p == -1:
            tampered[i] += 1.0
    panel.outcome = tampered
    result = did_estimate(panel)
    assert not result.parallel_trends_ok
