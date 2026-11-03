"""Propensity-score estimation: logistic regression from scratch.

No sklearn/statsmodels dependency — gradient descent on the
negative log-likelihood with L2 regularization, features standardized
internally so the optimizer is well conditioned. Standardization
parameters are stored so scoring new rows is consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 1.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var) or 1.0


@dataclass
class LogisticRegression:
    """Binary logistic regression fitted by batch gradient descent."""

    learning_rate: float = 0.1
    iterations: int = 1000
    l2: float = 0.01
    weights: list[float] = field(default_factory=list)
    intercept: float = 0.0
    feature_means: list[float] = field(default_factory=list)
    feature_stds: list[float] = field(default_factory=list)

    def _standardize(self, row: list[float]) -> list[float]:
        return [
            (v - m) / s
            for v, m, s in zip(row, self.feature_means, self.feature_stds)
        ]

    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticRegression":
        n = len(X)
        p = len(X[0]) if X else 0
        self.feature_means = [_mean([row[j] for row in X]) for j in range(p)]
        self.feature_stds = [
            _std([row[j] for row in X], self.feature_means[j]) for j in range(p)
        ]
        Xs = [self._standardize(row) for row in X]
        self.weights = [0.0] * p
        self.intercept = 0.0
        for _ in range(self.iterations):
            grad_w = [0.0] * p
            grad_b = 0.0
            for row, target in zip(Xs, y):
                z = self.intercept + sum(w * v for w, v in zip(self.weights, row))
                err = _sigmoid(z) - target
                grad_b += err
                for j in range(p):
                    grad_w[j] += err * row[j]
            for j in range(p):
                reg = self.l2 * self.weights[j]
                self.weights[j] -= self.learning_rate * (grad_w[j] / n + reg)
            self.intercept -= self.learning_rate * grad_b / n
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [
            _sigmoid(
                self.intercept
                + sum(w * v for w, v in zip(self.weights, self._standardize(row)))
            )
            for row in X
        ]

    def logit(self, X: list[list[float]]) -> list[float]:
        """Log-odds of treatment — the scale matching is done on."""
        out = []
        for prob in self.predict_proba(X):
            prob = min(max(prob, 1e-6), 1.0 - 1e-6)
            out.append(math.log(prob / (1.0 - prob)))
        return out


def fit_propensity(
    X: list[list[float]], treatment: list[int], **kwargs: object
) -> tuple[LogisticRegression, list[float]]:
    """Fit a propensity model and return (model, propensity scores)."""
    model = LogisticRegression(**kwargs)  # type: ignore[arg-type]
    model.fit(X, treatment)
    return model, model.predict_proba(X)
