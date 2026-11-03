"""Versioned experiment registry with promotion gates.

An experiment is an immutable versioned record of one causal analysis:
method, estimate, confidence interval, balance diagnostics, and
sensitivity result. New versions supersede old ones; nothing is edited
in place, so a shipped decision can always be traced back to the exact
numbers that justified it.

Promotion is fail-closed: an experiment promotes only if every gate
passes — balance (|SMD| < 0.1), a confidence interval that excludes
zero, a minimum matched/analysis sample, and (when available) parallel
trends holding and a sensitivity critical Gamma above the floor. Any
failed gate lists its reason; the registry never promotes on a partial
pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PromotionGate:
    max_abs_smd: float = 0.1
    min_sample: int = 100
    require_ci_excludes_zero: bool = True
    require_parallel_trends: bool = True
    min_critical_gamma: float | None = None

    def evaluate(
        self,
        max_abs_smd_after: float | None,
        ci_excludes_zero: bool,
        sample_size: int,
        parallel_trends_ok: bool | None,
        critical_gamma: float | None,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if max_abs_smd_after is not None and max_abs_smd_after >= self.max_abs_smd:
            reasons.append(
                f"balance failed: max |SMD| {max_abs_smd_after:.3f} >= {self.max_abs_smd}"
            )
        if sample_size < self.min_sample:
            reasons.append(
                f"sample too small: {sample_size} < {self.min_sample}"
            )
        if self.require_ci_excludes_zero and not ci_excludes_zero:
            reasons.append("confidence interval includes zero")
        if self.require_parallel_trends and parallel_trends_ok is False:
            reasons.append("parallel-trends check failed")
        if (
            self.min_critical_gamma is not None
            and critical_gamma is not None
            and critical_gamma < self.min_critical_gamma
        ):
            reasons.append(
                f"sensitivity too fragile: critical Gamma {critical_gamma} "
                f"< {self.min_critical_gamma}"
            )
        return (not reasons, reasons)


@dataclass
class Experiment:
    name: str
    method: str
    estimate: float
    ci_lower: float
    ci_upper: float
    max_abs_smd_after: float | None = None
    sample_size: int = 0
    parallel_trends_ok: bool | None = None
    critical_gamma: float | None = None
    version: int = 1
    status: str = "DRAFT"  # DRAFT | PROMOTED | REJECTED
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    gate_reasons: list[str] = field(default_factory=list)

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci_lower > 0 or self.ci_upper < 0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "method": self.method,
            "estimate": self.estimate,
            "ci": [self.ci_lower, self.ci_upper],
            "max_abs_smd_after": self.max_abs_smd_after,
            "sample_size": self.sample_size,
            "parallel_trends_ok": self.parallel_trends_ok,
            "critical_gamma": self.critical_gamma,
            "version": self.version,
            "status": self.status,
            "created_at": self.created_at,
            "gate_reasons": self.gate_reasons,
        }


class ExperimentRegistry:
    """In-memory registry; versions are append-only per experiment name."""

    def __init__(self, gate: PromotionGate | None = None) -> None:
        self.gate = gate or PromotionGate()
        self._experiments: dict[str, list[Experiment]] = {}

    def register(self, experiment: Experiment) -> Experiment:
        versions = self._experiments.setdefault(experiment.name, [])
        experiment.version = len(versions) + 1
        versions.append(experiment)
        return experiment

    def promote(self, name: str, version: int | None = None) -> Experiment:
        versions = self._experiments.get(name)
        if not versions:
            raise KeyError(f"unknown experiment: {name}")
        exp = versions[-1] if version is None else versions[version - 1]
        passed, reasons = self.gate.evaluate(
            max_abs_smd_after=exp.max_abs_smd_after,
            ci_excludes_zero=exp.ci_excludes_zero,
            sample_size=exp.sample_size,
            parallel_trends_ok=exp.parallel_trends_ok,
            critical_gamma=exp.critical_gamma,
        )
        exp.gate_reasons = reasons
        exp.status = "PROMOTED" if passed else "REJECTED"
        return exp

    def latest(self, name: str) -> Experiment:
        return self._experiments[name][-1]

    def promoted(self) -> list[Experiment]:
        return [
            exp
            for versions in self._experiments.values()
            for exp in versions
            if exp.status == "PROMOTED"
        ]

    def history(self, name: str) -> list[Experiment]:
        return list(self._experiments.get(name, []))
