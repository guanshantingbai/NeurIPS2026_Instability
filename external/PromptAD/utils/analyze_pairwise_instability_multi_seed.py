#!/usr/bin/env python3
"""
Multi-seed extension for pairwise instability vs penalized inference analysis.

For each (dataset, category, k), aggregates metrics across seeds: mean / std / count.

Depends on:
  - baseline per_sample: {baseline_root}/**/CLS-*-seed{SEED}-per_sample.csv
  - infer instability:   {infer_root}/**/CLS-*-seed{SEED}-per_sample_instability.csv

Run from PromptAD root:
  python utils/analyze_pairwise_instability_multi_seed.py \\
    --seeds 111,222,333,444,555 \\
    --out-dir pairwise_penalty_analysis_multi_seed_out

Baseline CSVs must exist for every seed under --baseline-root (same paths as infer).
If result_round1 only has seed111, either add per_sample for other seeds there or use:
  --baseline-root result_offline_infer
so I_bin is computed from the same inference run as penalized scores (per seed).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

from utils.analyze_pairwise_instability_baseline_vs_penalized import (
    analyze_one_setting,
    _parse_setting_with_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=str, default="result_round1")
    parser.add_argument("--infer-root", type=str, default="result_offline_infer")
    parser.add_argument("--penalty-col", type=str, default="s_penalty_0p1")
    parser.add_argument(
        "--seeds",
        type=str,
        default="111,222,333,444,555",
        help="Comma-separated seeds (must match checkpoint / CSV names).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="mvtec,visa",
        help="Comma-separated: mvtec, visa",
    )
    parser.add_argument("--out-dir", type=str, default="pairwise_penalty_analysis_multi_seed_out")
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    seed_set = set(seeds)
    allowed_ds = {d.strip().lower() for d in args.datasets.split(",") if d.strip()}

    pattern = os.path.join(args.baseline_root, "*", "k_*", "csv", "CLS-*-per_sample.csv")
    baseline_paths = sorted(
        p
        for p in glob.glob(pattern)
        if "instability" not in os.path.basename(p)
    )

    rows: List[Dict] = []
    skipped: List[str] = []

    for bp in baseline_paths:
        parsed = _parse_setting_with_seed(bp)
        if parsed is None:
            continue
        ds, cat, k, seed = parsed
        if ds not in allowed_ds or seed not in seed_set:
            continue
        rel = os.path.relpath(bp, args.baseline_root)
        ip_inst = os.path.join(args.infer_root, rel).replace("-per_sample.csv", "-per_sample_instability.csv")
        if not os.path.isfile(ip_inst):
            skipped.append(f"missing infer: seed={seed} {ip_inst}")
            continue
        try:
            out = analyze_one_setting(bp, ip_inst, args.penalty_col)
        except Exception as e:
            skipped.append(f"seed={seed} {bp}: {e}")
            continue
        if out is None:
            skipped.append(f"degenerate: seed={seed} {bp}")
            continue
        rows.append(out)

    os.makedirs(args.out_dir, exist_ok=True)
    skip_path = os.path.join(args.out_dir, "skipped.txt")
    with open(skip_path, "w", encoding="utf-8") as f:
        f.write("\n".join(skipped) if skipped else "(none)")

    if not rows:
        print("No rows analyzed. See", skip_path)
        return

    df = pd.DataFrame(rows)
    long_path = os.path.join(args.out_dir, "setting_seed_level_table.csv")
    df.to_csv(long_path, index=False, float_format="%.8f")

    gcols = ["dataset", "category", "k"]
    metrics = [
        "AUROC_baseline",
        "AUROC_penalized",
        "delta_AUROC",
        "rho_base",
        "rho_pen",
        "error_high_base",
        "error_high_pen",
        "delta_error_high",
        "delta_error_low",
        "delta_error_mid",
    ]
    agg_dict = {}
    for m in metrics:
        if m not in df.columns:
            continue
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
        agg_dict[f"{m}_n"] = (m, "count")
    agg = df.groupby(gcols, as_index=False).agg(**agg_dict)
    agg_path = os.path.join(args.out_dir, "setting_level_seed_aggregate.csv")
    agg.to_csv(agg_path, index=False, float_format="%.8f")

    # Global: across all (setting, seed) observations
    def gv(col: str) -> str:
        v = df[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return "nan"
        return f"mean={np.mean(v):.8f}  std={np.std(v, ddof=1) if len(v) > 1 else 0.0:.8f}  n={len(v)}"

    # Global: across aggregated settings (one value per setting = mean over seeds)
    setting_means = df.groupby(gcols, as_index=False).agg(
        delta_AUROC_m=("delta_AUROC", "mean"),
        delta_error_high_m=("delta_error_high", "mean"),
    )
    lines = [
        "=== Multi-seed analysis ===",
        f"seeds requested: {seeds}",
        f"rows (dataset×class×k×seed): {len(df)}",
        f"unique settings (dataset×class×k): {len(setting_means)}",
        "",
        "--- Pooled over all runs (each seed×setting is one sample) ---",
        f"delta_AUROC: {gv('delta_AUROC')}",
        f"delta_error_high: {gv('delta_error_high')}",
        "",
        "--- Across settings (each setting = mean over its seeds) ---",
        f"delta_AUROC: mean={setting_means['delta_AUROC_m'].mean():.8f}  std={(setting_means['delta_AUROC_m'].std(ddof=1) if len(setting_means) > 1 else 0.0):.8f}",
        f"delta_error_high: mean={setting_means['delta_error_high_m'].mean():.8f}  std={(setting_means['delta_error_high_m'].std(ddof=1) if len(setting_means) > 1 else 0.0):.8f}",
        "",
        f"long table: {long_path}",
        f"aggregate (mean/std/n per setting): {agg_path}",
        f"skipped: {skip_path} ({len(skipped)} lines)",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(args.out_dir, "global_summary_multi_seed.txt"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
