"""Risk-coverage analysis helpers."""

from __future__ import annotations

from typing import Iterable, List, Tuple


def compute_risk_coverage_curve(
    confidences: Iterable[float], errors: Iterable[float]
) -> List[Tuple[float, float]]:
    pairs = sorted(zip(confidences, errors), key=lambda x: x[0], reverse=True)
    curve: List[Tuple[float, float]] = []
    running_error = 0.0
    for idx, (_, err) in enumerate(pairs, start=1):
        running_error += err
        coverage = idx / len(pairs)
        risk = running_error / idx
        curve.append((coverage, risk))
    return curve
