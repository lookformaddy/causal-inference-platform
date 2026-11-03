"""Optional FastAPI service over the core estimators.

Kept thin on purpose: the service validates a request, runs the same
pure-stdlib estimators as the library, and returns the same shapes.
Importing this module without FastAPI installed raises ImportError;
tests auto-skip in that case (CI installs FastAPI).
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - exercised only without FastAPI
    raise

from causal_inference.data import generate_observational
from causal_inference.matching import balance_report, matched_ate, nearest_neighbour_match
from causal_inference.propensity import fit_propensity
from causal_inference.weighting import ipw_ate

app = FastAPI(title="Causal Inference Platform")


class AnalysisRequest(BaseModel):
    n: int = Field(default=2000, ge=100, le=20000)
    true_effect: float = 0.8
    seed: int = 42


class AnalysisResponse(BaseModel):
    naive_difference: float
    matched_ate: float
    ipw_ate: float
    max_abs_smd_before: float
    max_abs_smd_after: float
    balanced: bool
    n_matched_pairs: int
    n_unmatched_treated: int


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    data = generate_observational(
        n=request.n, true_effect=request.true_effect, seed=request.seed
    )
    model, propensity = fit_propensity(data.covariates, data.treatment)
    match = nearest_neighbour_match(model.logit(data.covariates), data.treatment)
    report = balance_report(
        data.covariates, data.covariate_names, data.treatment, match.pairs
    )
    treated_mean = sum(
        data.outcome[i] for i in data.treated()
    ) / max(len(data.treated()), 1)
    control_mean = sum(
        data.outcome[i] for i in data.control()
    ) / max(len(data.control()), 1)
    return AnalysisResponse(
        naive_difference=treated_mean - control_mean,
        matched_ate=matched_ate(data.outcome, match.pairs),
        ipw_ate=ipw_ate(data.outcome, data.treatment, propensity).estimate,
        max_abs_smd_before=report.max_abs_smd_before,
        max_abs_smd_after=report.max_abs_smd_after,
        balanced=report.balanced,
        n_matched_pairs=len(match.pairs),
        n_unmatched_treated=len(match.unmatched_treated),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
