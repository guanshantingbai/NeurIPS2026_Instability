"""Selection failure metrics for near-equivalent candidates."""

from __future__ import annotations


def failure_rate(num_failures: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return num_failures / total


def delta_risk(risk_selected: float, risk_best: float) -> float:
    return risk_selected - risk_best
