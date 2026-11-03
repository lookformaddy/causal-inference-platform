"""Versioned experiment registry and fail-closed promotion gates."""

import pytest

from causal_inference import Experiment, ExperimentRegistry, PromotionGate


def _good_experiment() -> Experiment:
    return Experiment(
        name="program-effect",
        method="ipw",
        estimate=0.82,
        ci_lower=0.61,
        ci_upper=1.03,
        max_abs_smd_after=0.04,
        sample_size=2000,
        parallel_trends_ok=True,
        critical_gamma=2.0,
    )


def test_promotion_passes_when_all_gates_pass():
    registry = ExperimentRegistry()
    exp = registry.register(_good_experiment())
    promoted = registry.promote(exp.name)
    assert promoted.status == "PROMOTED"
    assert promoted.gate_reasons == []
    assert registry.promoted() == [promoted]


def test_promotion_rejects_unbalanced_experiment():
    registry = ExperimentRegistry()
    exp = registry.register(_good_experiment())
    exp.max_abs_smd_after = 0.35
    promoted = registry.promote(exp.name)
    assert promoted.status == "REJECTED"
    assert any("balance" in r for r in promoted.gate_reasons)


def test_promotion_rejects_ci_including_zero():
    registry = ExperimentRegistry()
    exp = registry.register(_good_experiment())
    exp.ci_lower, exp.ci_upper = -0.1, 0.9
    assert registry.promote(exp.name).status == "REJECTED"


def test_promotion_rejects_failed_parallel_trends():
    registry = ExperimentRegistry(gate=PromotionGate(require_parallel_trends=True))
    exp = registry.register(_good_experiment())
    exp.parallel_trends_ok = False
    assert registry.promote(exp.name).status == "REJECTED"


def test_versions_are_append_only():
    registry = ExperimentRegistry()
    registry.register(_good_experiment())
    second = registry.register(_good_experiment())
    assert second.version == 2
    assert len(registry.history("program-effect")) == 2
    assert registry.latest("program-effect").version == 2


def test_promote_unknown_experiment_raises():
    with pytest.raises(KeyError):
        ExperimentRegistry().promote("nope")
