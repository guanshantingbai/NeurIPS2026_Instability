"""Instability metrics and helper transformations."""

from __future__ import annotations

from typing import Iterable


def normalized_instability(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    vmax, vmin = max(vals), min(vals)
    if vmax == vmin:
        return 0.0
    return (vmax - vmin) / max(abs(vmax), 1e-8)


def instability_gap(a: float, b: float) -> float:
    return abs(a - b)
