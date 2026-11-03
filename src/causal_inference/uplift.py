"""Uplift / decile analysis (two-model / T-learner approach).

Two linear outcome models are fitted from scratch — one on treated units,
one on controls — via the normal equations with Gaussian elimination
(pure stdlib). Predicted uplift for a unit is mu1(x) - mu0(x). Units are
ranked into deciles by predicted uplift; within each decile the observed
treated-minus-control outcome difference is reported. A useful model
concentrates the effect in the top deciles; a flat table means the
"targeting" adds nothing over treating everyone.

This is a diagnostic for heterogeneous effects, not a claim that the
top decile *causes* the lift — decile differences are still
observational and inherit the same confounding caveats as the ATE.
"""

from __future__ import annotations

from dataclasses import dataclass


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system: cannot fit outcome model")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def _fit_ols(X: list[list[float]], y: list[float], ridge: float = 1e-6) -> list[float]:
    """Least squares with intercept; returns [intercept, w1, ...]."""
    p = len(X[0]) if X else 0
    dim = p + 1
    a = [[0.0] * dim for _ in range(dim)]
    b = [0.0] * dim
    for row, target in zip(X, y):
        feat = [1.0] + row
        for i in range(dim):
            b[i] += feat[i] * target
            for j in range(dim):
                a[i][j] += feat[i] * feat[j]
    for i in range(dim):
        a[i][i] += ridge
    return _solve_linear(a, b)


def _predict(coef: list[float], row: list[float]) -> float:
    return coef[0] + sum(w * v for w, v in zip(coef[1:], row))


@dataclass
class UpliftModel:
    treated_coef: list[float]
    control_coef: list[float]

    def predicted_uplift(self, X: list[list[float]]) -> list[float]:
        return [
            _predict(self.treated_coef, row) - _predict(self.control_coef, row)
            for row in X
        ]


def fit_uplift_model(
    X: list[list[float]], treatment: list[int], outcome: list[float]
) -> UpliftModel:
    """Fit separate outcome regressions per arm (T-learner)."""
    x_t = [row for row, t in zip(X, treatment) if t == 1]
    y_t = [y for y, t in zip(outcome, treatment) if t == 1]
    x_c = [row for row, t in zip(X, treatment) if t == 0]
    y_c = [y for y, t in zip(outcome, treatment) if t == 0]
    if len(x_t) < 2 or len(x_c) < 2:
        raise ValueError("both arms need at least 2 observations")
    return UpliftModel(
        treated_coef=_fit_ols(x_t, y_t), control_coef=_fit_ols(x_c, y_c)
    )


@dataclass
class DecileRow:
    decile: int  # 1 = lowest predicted uplift, 10 = highest
    n_treated: int
    n_control: int
    mean_treated: float
    mean_control: float
    observed_uplift: float


def decile_uplift(
    predicted_uplift: list[float],
    treatment: list[int],
    outcome: list[float],
    n_bins: int = 10,
) -> list[DecileRow]:
    """Rank units into equal-sized bins by predicted uplift and report
    the observed treated-minus-control difference within each bin."""
    order = sorted(range(len(predicted_uplift)), key=lambda i: predicted_uplift[i])
    n = len(order)
    rows: list[DecileRow] = []
    for b in range(n_bins):
        start = b * n // n_bins
        end = (b + 1) * n // n_bins
        members = order[start:end]
        t_vals = [outcome[i] for i in members if treatment[i] == 1]
        c_vals = [outcome[i] for i in members if treatment[i] == 0]
        mt = sum(t_vals) / len(t_vals) if t_vals else 0.0
        mc = sum(c_vals) / len(c_vals) if c_vals else 0.0
        rows.append(
            DecileRow(
                decile=b + 1,
                n_treated=len(t_vals),
                n_control=len(c_vals),
                mean_treated=mt,
                mean_control=mc,
                observed_uplift=mt - mc,
            )
        )
    return rows
