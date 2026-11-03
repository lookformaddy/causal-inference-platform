"""Causal inference platform: propensity models, matching, weighting,
difference-in-differences, uplift, bootstrap inference, sensitivity analysis,
and a versioned experiment registry. Pure standard library."""

from causal_inference.data import PanelData, SyntheticData, generate_observational, generate_panel
from causal_inference.propensity import LogisticRegression, fit_propensity
from causal_inference.matching import BalanceReport, matched_ate, nearest_neighbour_match, standardized_mean_difference
from causal_inference.weighting import ipw_ate
from causal_inference.did import did_estimate, parallel_trends_check
from causal_inference.uplift import decile_uplift, fit_uplift_model
from causal_inference.inference import bootstrap_ci
from causal_inference.sensitivity import rosenbaum_bounds
from causal_inference.registry import Experiment, ExperimentRegistry, PromotionGate

__all__ = [
    "PanelData",
    "SyntheticData",
    "generate_observational",
    "generate_panel",
    "LogisticRegression",
    "fit_propensity",
    "BalanceReport",
    "matched_ate",
    "nearest_neighbour_match",
    "standardized_mean_difference",
    "ipw_ate",
    "did_estimate",
    "parallel_trends_check",
    "decile_uplift",
    "fit_uplift_model",
    "bootstrap_ci",
    "rosenbaum_bounds",
    "Experiment",
    "ExperimentRegistry",
    "PromotionGate",
]
