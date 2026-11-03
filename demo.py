"""End-to-end demo: recover a known true effect from confounded data.

Runs the full pipeline on synthetic data with true_effect = 0.8:
naive difference (biased) -> propensity model -> matching with balance
diagnostics -> IPW -> bootstrap CI -> sensitivity bounds -> DiD on a
panel with true_effect = 0.6 -> uplift deciles -> registry promotion
gate. Asserts the adjusted estimates land near the truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from causal_inference import (  # noqa: E402
    Experiment,
    ExperimentRegistry,
    bootstrap_ci,
    decile_uplift,
    did_estimate,
    fit_propensity,
    fit_uplift_model,
    generate_observational,
    generate_panel,
    ipw_ate,
    matched_ate,
    nearest_neighbour_match,
    rosenbaum_bounds,
)
from causal_inference.matching import balance_report  # noqa: E402


def main() -> None:
    print("== Causal Inference Platform demo ==\n")

    # 1. Observational data with a known true effect of 0.8.
    data = generate_observational(n=2000, true_effect=0.8, seed=42)
    naive = sum(data.outcome[i] for i in data.treated()) / len(data.treated())
    naive -= sum(data.outcome[i] for i in data.control()) / len(data.control())
    print(f"True effect:            {data.true_effect:.3f}")
    print(f"Naive difference:       {naive:.3f}  (biased by confounding)")

    # 2. Propensity model + matching + balance diagnostics.
    model, propensity = fit_propensity(data.covariates, data.treatment)
    match = nearest_neighbour_match(model.logit(data.covariates), data.treatment)
    report = balance_report(
        data.covariates, data.covariate_names, data.treatment, match.pairs
    )
    m_ate = matched_ate(data.outcome, match.pairs)
    print(f"Matched pairs:          {len(match.pairs)} "
          f"({len(match.unmatched_treated)} treated unmatched, outside caliper)")
    print(f"Max |SMD| before:       {report.max_abs_smd_before:.3f}")
    print(f"Max |SMD| after:        {report.max_abs_smd_after:.3f} "
          f"(balanced={report.balanced})")
    print(f"Matched ATE:            {m_ate:.3f}")

    # 3. IPW.
    ipw = ipw_ate(data.outcome, data.treatment, propensity)
    print(f"IPW ATE:                {ipw.estimate:.3f} "
          f"(max weight {ipw.max_weight:.1f}, {ipw.n_clipped} scores clipped)")

    # 4. Bootstrap CI around the IPW estimate.
    def ipw_on_sample(idx: list[int]) -> float:
        # Resample rows, holding propensity scores fixed (standard
        # approximation; refitting per replicate is supported by the API).
        return ipw_ate(
            [data.outcome[i] for i in idx],
            [data.treatment[i] for i in idx],
            [propensity[i] for i in idx],
        ).estimate

    ci = bootstrap_ci(ipw_on_sample, data.n, ipw.estimate, n_bootstraps=200)
    print(f"IPW 95% CI:             [{ci.lower:.3f}, {ci.upper:.3f}] "
          f"(excludes 0: {ci.excludes_zero})")

    # 5. Sensitivity: how much hidden bias would explain the effect away?
    sens = rosenbaum_bounds(data.outcome, match.pairs)
    print(f"Rosenbaum critical Gamma: {sens.critical_gamma} "
          f"(pairs={sens.n_pairs}, positive={sens.n_positive})")

    # 6. Difference-in-differences on panel data (true effect 0.6).
    panel = generate_panel(n_units=800, true_effect=0.6, seed=7)
    did = did_estimate(panel)
    print(f"\nDiD estimate:           {did.estimate:.3f} (true {panel.true_effect})")
    print(f"Placebo DiD (pre only): {did.placebo_estimate:.3f} "
          f"(parallel trends ok: {did.parallel_trends_ok})")

    # 7. Uplift deciles.
    uplift_model = fit_uplift_model(data.covariates, data.treatment, data.outcome)
    predicted = uplift_model.predicted_uplift(data.covariates)
    deciles = decile_uplift(predicted, data.treatment, data.outcome)
    print("\nUplift deciles (1 = lowest predicted uplift):")
    for row in deciles:
        print(f"  D{row.decile:>2}: n_t={row.n_treated:>3} n_c={row.n_control:>3} "
              f"observed uplift {row.observed_uplift:+.3f}")

    # 8. Registry promotion gate.
    registry = ExperimentRegistry()
    exp = registry.register(
        Experiment(
            name="program-effect-observational",
            method="ipw",
            estimate=ipw.estimate,
            ci_lower=ci.lower,
            ci_upper=ci.upper,
            max_abs_smd_after=report.max_abs_smd_after,
            sample_size=data.n,
            parallel_trends_ok=did.parallel_trends_ok,
            critical_gamma=sens.critical_gamma,
        )
    )
    registry.promote(exp.name)
    print(f"\nRegistry: {exp.name} v{exp.version} -> {exp.status}"
          + (f" ({'; '.join(exp.gate_reasons)})" if exp.gate_reasons else ""))

    # Verification: adjusted estimates must recover the planted effect.
    assert abs(m_ate - 0.8) < 0.25, f"matched ATE off: {m_ate}"
    assert abs(ipw.estimate - 0.8) < 0.25, f"IPW off: {ipw.estimate}"
    assert ci.lower <= 0.8 <= ci.upper, "CI misses the true effect"
    assert abs(did.estimate - 0.6) < 0.25, f"DiD off: {did.estimate}"
    assert did.parallel_trends_ok, "parallel trends should hold on synthetic panel"
    print("\nDemo OK: adjusted estimates recover the planted effects.")


if __name__ == "__main__":
    main()
