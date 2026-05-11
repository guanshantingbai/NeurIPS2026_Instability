#!/usr/bin/env python3
"""
Build a single paper-ready CSV: one row per (dataset, class, k, seed) with AUROC + instability.

Sources (later overwrites earlier on duplicate keys):
  1) ``result_round1/**/pairwise_instability/CLS-*-seed*-per_sample/summary.json``
     — typically **seed 111** for the full 81-setting grid.
  2) ``result_seed_search/{slug}/{seed}/pairwise_instability/summary.json``
     — multi-seed re-runs (e.g. 111–555) for settings that were trained under seed search.

``summary.json`` fields used:
  - ``sklearn_auroc_final`` → column ``auroc``
  - ``flip_rate_mean`` → column ``instability``

Usage (from repo root)::

  python PromptAD/utils/build_per_seed_setting_metrics_csv.py \\
      --out-csv PromptAD/result_analysis/per_seed_setting_metrics.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROMPTAD_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROMPTAD_ROOT not in sys.path:
    sys.path.insert(0, _PROMPTAD_ROOT)

from utils.seed_killer_evidence_pipeline import (  # noqa: E402
    discover_pairwise_runs,
    load_json,
)


def _parse_seed_from_run_name(run_name: str) -> Optional[int]:
    m = re.search(r"-seed(\d+)-per_sample$", run_name)
    return int(m.group(1)) if m else None


def _iter_round1_rows(result_round1: str) -> Iterator[Dict[str, Any]]:
    for run in discover_pairwise_runs(os.path.abspath(result_round1)):
        run_name = str(run["run_name"])
        seed = _parse_seed_from_run_name(run_name)
        if seed is None:
            continue
        summ_path = run["summary_path"]
        if not os.path.isfile(summ_path):
            continue
        s = load_json(summ_path)
        yield {
            "dataset": str(run["dataset"]),
            "class": str(run["category"]),
            "k": int(run["k"]),
            "seed": int(seed),
            "auroc": float(s.get("sklearn_auroc_final", float("nan"))),
            "instability": float(s.get("flip_rate_mean", float("nan"))),
            "num_samples": int(s.get("num_samples", -1)),
            "num_pairs": int(s.get("num_pairs", -1)),
            "run_name": run_name,
            "summary_path": os.path.abspath(summ_path),
            "data_source": "result_round1",
        }


_SLUG_RE = re.compile(r"^(?P<dataset>mvtec|visa)__(?P<class>.+)__k(?P<k>\d+)$")


def _parse_slug(slug: str) -> Optional[Tuple[str, str, int]]:
    m = _SLUG_RE.match(slug)
    if not m:
        return None
    return m.group("dataset"), m.group("class"), int(m.group("k"))


def _iter_seed_search_rows(result_seed_search: str) -> Iterator[Dict[str, Any]]:
    root = os.path.abspath(result_seed_search)
    pattern = os.path.join(root, "*", "*", "pairwise_instability", "summary.json")
    for summ_path in sorted(glob.glob(pattern)):
        parts = summ_path.split(os.sep)
        # .../slug/seed/pairwise_instability/summary.json
        if len(parts) < 5:
            continue
        seed_dir = parts[-3]
        slug = parts[-4]
        if not seed_dir.isdigit():
            continue
        seed = int(seed_dir)
        parsed = _parse_slug(slug)
        if parsed is None:
            continue
        dataset, cls, k = parsed
        if not os.path.isfile(summ_path):
            continue
        s = load_json(summ_path)
        yield {
            "dataset": dataset,
            "class": cls,
            "k": k,
            "seed": seed,
            "auroc": float(s.get("sklearn_auroc_final", float("nan"))),
            "instability": float(s.get("flip_rate_mean", float("nan"))),
            "num_samples": int(s.get("num_samples", -1)),
            "num_pairs": int(s.get("num_pairs", -1)),
            "run_name": slug,
            "summary_path": os.path.abspath(summ_path),
            "data_source": "result_seed_search",
        }


def build_unified_table(
    result_round1: str,
    result_seed_search: str,
) -> pd.DataFrame:
    """Later rows overwrite earlier rows on (dataset, class, k, seed)."""
    rows: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
    order: List[Tuple[str, str, int, int]] = []

    def _add(r: Dict[str, Any]) -> None:
        key = (r["dataset"], r["class"], int(r["k"]), int(r["seed"]))
        if key not in rows:
            order.append(key)
        rows[key] = r

    for r in _iter_round1_rows(result_round1):
        _add(dict(r))
    for r in _iter_seed_search_rows(result_seed_search):
        _add(dict(r))

    out = [rows[k] for k in order]
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df = df.sort_values(["dataset", "class", "k", "seed"], kind="mergesort").reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--result-round1",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_round1"),
    )
    p.add_argument(
        "--result-seed-search",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search"),
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_analysis", "per_seed_setting_metrics.csv"),
    )
    p.add_argument(
        "--out-manifest",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_analysis", "per_seed_setting_metrics_manifest.json"),
    )
    args = p.parse_args()

    df = build_unified_table(args.result_round1, args.result_seed_search)
    if df.empty:
        raise SystemExit("No rows collected; check result_round1 / result_seed_search paths.")

    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    # Paper-facing core columns first
    core = df[["dataset", "class", "k", "seed", "auroc", "instability"]].copy()
    extra = df[["num_samples", "num_pairs", "data_source", "run_name", "summary_path"]]
    out_df = pd.concat([core, extra], axis=1)
    out_df.to_csv(out_csv, index=False)

    src_counts = df.groupby("data_source").size().to_dict()
    manifest = {
        "out_csv": out_csv,
        "n_rows": int(len(df)),
        "n_unique_settings": int(df.groupby(["dataset", "class", "k"]).ngroups),
        "rows_by_source": {str(k): int(v) for k, v in src_counts.items()},
        "seeds_present": sorted(df["seed"].unique().tolist()),
        "note": "Duplicate (dataset, class, k, seed): result_seed_search overwrites result_round1.",
    }
    man_path = os.path.abspath(args.out_manifest)
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_csv} ({len(out_df)} rows)")
    print(f"Wrote {man_path}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
