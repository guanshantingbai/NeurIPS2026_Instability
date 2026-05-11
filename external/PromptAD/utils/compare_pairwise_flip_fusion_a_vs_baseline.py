#!/usr/bin/env python3
"""
Compare pairwise instability (flip rate / I_bin) between:
  - baseline: z_final from harmonic s_final (s_final_baseline in fusion_a CSV)
  - Strategy A: z_final from s_final_instability_fusion

Uses the same definitions as utils.pairwise_instability.build_pairwise_table:
  flip = 1 iff z_sem, z_vis, z_final are not all identical
  flip_rate_mean = mean(flip) over all anomaly-normal pairs
  I_bin = per-image mean flip over paired opposite-label samples

Also optionally loads result_round1/**/summary.json flip_rate_mean for sanity check
(baseline should match recomputation from s_final_baseline when tables align).

Run from PromptAD repo root:
  python utils/compare_pairwise_flip_fusion_a_vs_baseline.py \\
    --fusion-root result_offline_infer \\
    --baseline-summaries-root result_round1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from utils.pairwise_instability import build_pairwise_table, build_sample_instability_table


def _fusion_csv_to_score_df(path: str, final_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    lab = pd.to_numeric(df["label"], errors="raise").astype(int)
    return pd.DataFrame(
        {
            "image_path": df["image_path"].astype(str),
            "image_label": np.where(lab == 1, "anomaly", "normal"),
            "semantic_score": pd.to_numeric(df["s_sem"], errors="raise"),
            "visual_score": pd.to_numeric(df["s_vis"], errors="raise"),
            "harmonic_score": pd.to_numeric(df[final_col], errors="raise"),
        }
    )


def _rel_key_from_fusion_path(fusion_root: str, path: str) -> str:
    rel = os.path.relpath(path, fusion_root)
    # .../mvtec/k_1/csv/CLS-mvtec-carpet-k1-seed111-per_sample_instability_fusion_a.csv
    # -> mvtec/k_1/pairwise_instability/CLS-mvtec-carpet-k1-seed111-per_sample/summary.json
    parts = rel.split(os.sep)
    if len(parts) < 4 or parts[-2] != "csv":
        return ""
    ds, kpart = parts[0], parts[1]
    base = os.path.basename(path)
    stem = base.replace("-per_sample_instability_fusion_a.csv", "-per_sample")
    return os.path.join(ds, kpart, "pairwise_instability", stem, "summary.json")


def _load_round1_flip_summary(baseline_root: str, rel_key: str) -> Optional[float]:
    p = os.path.join(baseline_root, rel_key)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return float(d["flip_rate_mean"])


def analyze_one_fusion_csv(path: str) -> Dict[str, Any]:
    df_b = _fusion_csv_to_score_df(path, "s_final_baseline")
    df_f = _fusion_csv_to_score_df(path, "s_final_instability_fusion")

    pair_b = build_pairwise_table(df_b)
    pair_f = build_pairwise_table(df_f)

    fr_b = float(pair_b["flip"].mean())
    fr_f = float(pair_f["flip"].mean())

    samp_b = build_sample_instability_table(pair_b)
    samp_f = build_sample_instability_table(pair_f)
    merged = samp_b.merge(
        samp_f,
        on=["image_path", "image_label"],
        suffixes=("_base", "_fus"),
    )
    merged["delta_I_bin"] = merged["I_bin_fus"] - merged["I_bin_base"]

    return {
        "path": path,
        "num_pairs": int(len(pair_b)),
        "flip_rate_baseline": fr_b,
        "flip_rate_fusion_a": fr_f,
        "delta_flip_rate": fr_f - fr_b,
        "mean_I_bin_baseline": float(samp_b["I_bin"].mean()),
        "mean_I_bin_fusion_a": float(samp_f["I_bin"].mean()),
        "mean_delta_I_bin_per_sample": float(merged["delta_I_bin"].mean()),
        "median_delta_I_bin_per_sample": float(merged["delta_I_bin"].median()),
        "frac_samples_lower_flip_instability": float((merged["delta_I_bin"] < 0).mean()),
        "frac_samples_equal": float((merged["delta_I_bin"] == 0).mean()),
        "frac_samples_higher": float((merged["delta_I_bin"] > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fusion-root",
        type=str,
        default="result_offline_infer",
        help="Root containing *per_sample_instability_fusion_a.csv",
    )
    parser.add_argument(
        "--baseline-summaries-root",
        type=str,
        default="result_round1",
        help="Root containing precomputed pairwise_instability/**/summary.json",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional path to write per-setting table",
    )
    args = parser.parse_args()

    fusion_root = os.path.abspath(args.fusion_root)
    base_root = os.path.abspath(args.baseline_summaries_root)

    pattern = os.path.join(fusion_root, "**", "CLS-*-per_sample_instability_fusion_a.csv")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        print(f"No fusion_a CSV under {fusion_root}", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    cmp_ok = 0
    cmp_mismatch = 0
    cmp_missing = 0
    max_abs_diff = 0.0

    for p in paths:
        r = analyze_one_fusion_csv(p)
        rel = _rel_key_from_fusion_path(fusion_root, p)
        json_fr = _load_round1_flip_summary(base_root, rel) if rel else None
        r["round1_summary_flip_rate"] = json_fr
        if json_fr is None:
            cmp_missing += 1
            r["baseline_vs_round1_abs_diff"] = float("nan")
        else:
            d = abs(r["flip_rate_baseline"] - json_fr)
            max_abs_diff = max(max_abs_diff, d)
            r["baseline_vs_round1_abs_diff"] = d
            if d < 1e-9:
                cmp_ok += 1
            else:
                cmp_mismatch += 1
        rows.append(r)

    tbl = pd.DataFrame(rows)

    print(f"Settings (fusion_a files): {len(paths)}")
    print()
    print("=== Pairwise flip rate (mean over all anomaly–normal pairs) ===")
    print(f"  Baseline (z_final = harmonic / s_final_baseline):  mean={tbl['flip_rate_baseline'].mean():.6f}, median={tbl['flip_rate_baseline'].median():.6f}")
    print(f"  Strategy A (z_final = s_final_instability_fusion): mean={tbl['flip_rate_fusion_a'].mean():.6f}, median={tbl['flip_rate_fusion_a'].median():.6f}")
    print(f"  Delta (fusion - baseline), mean over settings:   {tbl['delta_flip_rate'].mean():+.6f}")
    print(f"  Delta (fusion - baseline), median over settings:   {tbl['delta_flip_rate'].median():+.6f}")
    print(f"  Count settings with lower / equal / higher flip rate: "
          f"{int((tbl['delta_flip_rate'] < 0).sum())} / {int((tbl['delta_flip_rate'] == 0).sum())} / {int((tbl['delta_flip_rate'] > 0).sum())}")
    print()
    print("=== Sample-level I_bin (mean over paired opposite-label flips) ===")
    print(f"  Mean I_bin baseline: {tbl['mean_I_bin_baseline'].mean():.6f}")
    print(f"  Mean I_bin fusion A: {tbl['mean_I_bin_fusion_a'].mean():.6f}")
    print(f"  Mean per-sample delta I_bin (fusion - baseline), averaged over settings: {tbl['mean_delta_I_bin_per_sample'].mean():+.6f}")
    print(f"  Fraction of samples (pooled per-setting means) with delta_I_bin < 0 / =0 / >0: "
          f"{tbl['frac_samples_lower_flip_instability'].mean():.4f} / {tbl['frac_samples_equal'].mean():.4f} / {tbl['frac_samples_higher'].mean():.4f}")
    print()
    print("=== Sanity: recomputed baseline flip vs result_round1 summary.json ===")
    print(f"  Matched summary.json: {cmp_ok}, mismatch: {cmp_mismatch}, missing json: {cmp_missing}")
    if cmp_mismatch == 0 and cmp_missing == 0:
        print(f"  max |flip_recomputed - flip_round1|: {max_abs_diff:.3e}")
    elif cmp_mismatch:
        print(f"  max |flip_recomputed - flip_round1| (among compared): {max_abs_diff:.6e}")
        print("  [WARN] Mismatches may mean per_sample / fusion export differs from round1 table.")

    if args.out_csv.strip():
        out = os.path.abspath(args.out_csv.strip())
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        tbl.to_csv(out, index=False, float_format="%.8f")
        print()
        print(f"[saved] {out}")


if __name__ == "__main__":
    os.chdir(_REPO)
    main()
