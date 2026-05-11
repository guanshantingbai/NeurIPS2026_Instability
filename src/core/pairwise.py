"""Pairwise utilities shared across section pipelines."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Sequence, Tuple


def build_pairs(items: Sequence[str]) -> List[Tuple[str, str]]:
    """Return all unordered pairs from identifiers."""
    return list(combinations(items, 2))


def filter_near_metric_pairs(
    pairs: Iterable[Tuple[str, str]],
    metric_lookup: dict[str, float],
    max_gap: float,
) -> List[Tuple[str, str]]:
    """Keep pairs whose absolute metric gap is below threshold."""
    kept: List[Tuple[str, str]] = []
    for left, right in pairs:
        if abs(metric_lookup[left] - metric_lookup[right]) <= max_gap:
            kept.append((left, right))
    return kept
