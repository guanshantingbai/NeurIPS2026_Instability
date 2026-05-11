#!/usr/bin/env python3
"""
Emit one TSV line per (dataset, class, k, seed) that lacks
``{search_root}/{slug}/{seed}/pairwise_instability/summary.json``.

Settings are taken from ``result_round1`` pairwise runs (same 81-grid as round1).

TSV columns: dataset, class, k, seed, slug, job_root

Usage:
  python PromptAD/utils/emit_missing_promptad_seed_jobs.py \\
      --seeds 222,333,444,555 \\
      --search-root PromptAD/result_seed_search
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Set, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROMPTAD_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROMPTAD_ROOT not in sys.path:
    sys.path.insert(0, _PROMPTAD_ROOT)

from utils.seed_killer_evidence_pipeline import discover_pairwise_runs  # noqa: E402


def _parse_seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in s.replace(",", " ").split():
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


def _unique_settings(result_round1: str) -> List[Tuple[str, str, int]]:
    seen: Set[Tuple[str, str, int]] = set()
    rows: List[Tuple[str, str, int]] = []
    for run in discover_pairwise_runs(os.path.abspath(result_round1)):
        key = (str(run["dataset"]), str(run["category"]), int(run["k"]))
        if key not in seen:
            seen.add(key)
            rows.append(key)
    rows.sort(key=lambda t: (t[0], t[1], t[2]))
    return rows


def emit_lines(
    result_round1: str,
    search_root: str,
    seeds: Iterable[int],
) -> List[str]:
    lines: List[str] = []
    root = os.path.abspath(search_root)
    for dataset, cls, k in _unique_settings(result_round1):
        slug = f"{dataset}__{cls}__k{k}"
        for seed in seeds:
            summ = os.path.join(root, slug, str(int(seed)), "pairwise_instability", "summary.json")
            if os.path.isfile(summ):
                continue
            job_root = os.path.join(root, slug, str(int(seed)))
            lines.append(
                "\t".join(
                    [
                        dataset,
                        cls,
                        str(int(k)),
                        str(int(seed)),
                        slug,
                        job_root,
                    ]
                )
            )
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--result-round1",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_round1"),
    )
    p.add_argument(
        "--search-root",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search"),
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="222,333,444,555",
        help="Comma/space-separated seeds to fill under result_seed_search (default skips 111).",
    )
    args = p.parse_args()
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seeds after --seeds parse.")

    for line in emit_lines(args.result_round1, args.search_root, seeds):
        print(line)


if __name__ == "__main__":
    main()
