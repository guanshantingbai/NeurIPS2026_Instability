#!/usr/bin/env python3
"""
Compare baseline (result_round1 per_sample) vs penalized inference (result_offline_infer).

Uses **pairwise instability I_bin** from utils.pairwise_instability (flip rate over
opposite-label pairs; semantic vs visual vs harmonic agreement) — not u_abs/u_norm.

Reads:
  baseline:  {baseline_root}/**/csv/*-per_sample.csv  (excludes *instability*)
  penalized: infer side CSV column, e.g. s_penalty_0p1 in *per_sample_instability.csv
  (H2/thresholded), or s_final_instability_fusion in *per_sample_instability_fusion_a.csv
  (Strategy A weighted harmonic).

Run from PromptAD repo root:
  python utils/analyze_pairwise_instability_baseline_vs_penalized.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from utils.instability_penalty import compute_ranking_error
from utils.pairwise_instability import build_pairwise_table, build_sample_instability_table, load_score_table


def _instability_path_from_per_sample(per_sample_path: str) -> str:
    return per_sample_path.replace("-per_sample.csv", "-per_sample_instability.csv")


def _parse_setting_with_seed(path: str) -> Optional[Tuple[str, str, int, int]]:
    """
    CLS-{dataset}-{class}-k{k}-seed{seed}-per_sample.csv
    Returns (dataset, category, k_shot, seed).
    """
    base = os.path.basename(path)
    m = re.match(
        r"^CLS-(mvtec|visa)-(.+)-k(\d+)-seed(\d+)-per_sample\.csv$",
        base,
    )
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3)), int(m.group(4))


def _parse_setting(path: str) -> Optional[Tuple[str, str, int]]:
    p = _parse_setting_with_seed(path)
    if p is None:
        return None
    return p[0], p[1], p[2]


def _bucket_masks(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """low bottom 30%, mid middle 40%, high top 30% by percentile rank."""
    s = pd.Series(I)
    pct = s.rank(pct=True, method="average").to_numpy(dtype=float)
    low = pct <= 0.30
    high = pct > 0.70
    mid = (pct > 0.30) & (pct <= 0.70)
    return low, mid, high


def analyze_one_setting(
    baseline_csv: str,
    infer_instability_csv: str,
    penalty_col: str,
) -> Optional[Dict[str, float]]:
    df_base = load_score_table(baseline_csv)
    pair_df = build_pairwise_table(df_base)
    samp = build_sample_instability_table(pair_df)
    I_map = samp.set_index("image_path")["I_bin"]

    if not os.path.isfile(infer_instability_csv):
        return None
    df_pen = pd.read_csv(infer_instability_csv)
    if penalty_col not in df_pen.columns:
        raise ValueError(f"Missing {penalty_col} in {infer_instability_csv}")

    # Align on image_path
    df = df_base[["image_path", "image_label"]].copy()
    df["score_baseline"] = pd.to_numeric(df_base["harmonic_score"], errors="raise")
    m = df["image_path"].map(df_pen.set_index("image_path")[penalty_col])
    if m.isna().any():
        missing = int(m.isna().sum())
        raise ValueError(f"{infer_instability_csv}: {missing} rows missing in penalized table")
    df["score_penalized"] = pd.to_numeric(m, errors="raise")

    df["I_base"] = df["image_path"].map(I_map)
    if df["I_base"].isna().any():
        raise ValueError("I_base merge failed for some paths")

    y = df["image_label"].map({"normal": 0, "anomaly": 1}).to_numpy(dtype=int)
    s_b = df["score_baseline"].to_numpy(dtype=float)
    s_p = df["score_penalized"].to_numpy(dtype=float)
    I = df["I_base"].to_numpy(dtype=float)

    e_base = compute_ranking_error(s_b, y)
    e_pen = compute_ranking_error(s_p, y)

    mask = np.isfinite(e_base) & np.isfinite(e_pen)
    if mask.sum() < 3:
        return None

    rho_b, _ = spearmanr(I[mask], e_base[mask])
    rho_p, _ = spearmanr(I[mask], e_pen[mask])
    rho_b = float(rho_b) if rho_b == rho_b else float("nan")
    rho_p = float(rho_p) if rho_p == rho_p else float("nan")

    low, mid, high = _bucket_masks(I)

    def mean_err(mk, e):
        if not np.any(mk):
            return float("nan")
        return float(np.mean(e[mk & np.isfinite(e)]))

    eb_l, eb_m, eb_h = mean_err(low, e_base), mean_err(mid, e_base), mean_err(high, e_base)
    ep_l, ep_m, ep_h = mean_err(low, e_pen), mean_err(mid, e_pen), mean_err(high, e_pen)

    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        auroc_b = auroc_p = float("nan")
    else:
        auroc_b = float(roc_auc_score(y, s_b))
        auroc_p = float(roc_auc_score(y, s_p))

    parsed = _parse_setting_with_seed(baseline_csv)
    if parsed is None:
        return None
    dataset, category, k, seed = parsed

    return {
        "dataset": dataset,
        "category": category,
        "k": k,
        "seed": int(seed),
        "AUROC_baseline": auroc_b,
        "AUROC_penalized": auroc_p,
        "delta_AUROC": auroc_p - auroc_b,
        "rho_base": rho_b,
        "rho_pen": rho_p,
        "error_low_base": eb_l,
        "error_mid_base": eb_m,
        "error_high_base": eb_h,
        "error_low_pen": ep_l,
        "error_mid_pen": ep_m,
        "error_high_pen": ep_h,
        "delta_error_low": ep_l - eb_l,
        "delta_error_mid": ep_m - eb_m,
        "delta_error_high": ep_h - eb_h,
        "n_samples": float(len(df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=str, default="result_round1")
    parser.add_argument("--infer-root", type=str, default="result_offline_infer")
    parser.add_argument("--penalty-col", type=str, default="s_penalty_0p1")
    parser.add_argument(
        "--datasets",
        type=str,
        default="mvtec,visa",
        help="Comma-separated subset: mvtec, visa (default: both).",
    )
    parser.add_argument("--out-dir", type=str, default="pairwise_penalty_analysis_out")
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)

    allowed_ds = {d.strip().lower() for d in args.datasets.split(",") if d.strip()}

    pattern = os.path.join(args.baseline_root, "*", "k_*", "csv", "CLS-*-per_sample.csv")
    baseline_paths = sorted(
        p
        for p in glob.glob(pattern)
        if "instability" not in os.path.basename(p)
    )
    filtered: List[str] = []
    for bp in baseline_paths:
        parsed = _parse_setting(bp)
        if parsed is None:
            continue
        if parsed[0] in allowed_ds:
            filtered.append(bp)
    baseline_paths = filtered

    rows: List[Dict[str, float]] = []
    skipped: List[str] = []

    for bp in baseline_paths:
        rel = os.path.relpath(bp, args.baseline_root)
        ip_inst = os.path.join(args.infer_root, rel).replace("-per_sample.csv", "-per_sample_instability.csv")
        if not os.path.isfile(ip_inst):
            skipped.append(f"missing infer instability: {ip_inst}")
            continue
        try:
            out = analyze_one_setting(bp, ip_inst, args.penalty_col)
        except Exception as e:
            skipped.append(f"{bp}: {e}")
            continue
        if out is None:
            skipped.append(f"skip (degenerate): {bp}")
            continue
        rows.append(out)

    if not rows:
        print("No settings analyzed.")
        print(f"  (datasets filter: {sorted(allowed_ds)})")
        for s in skipped[:25]:
            print(" ", s)
        return

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    detail_path = os.path.join(args.out_dir, "setting_level_table.csv")
    df.to_csv(detail_path, index=False, float_format="%.6f")

    # Paper-style compact table: class | k | ...
    mv = df[df["dataset"] == "mvtec"].copy()
    slim = mv[
        [
            "category",
            "k",
            "delta_AUROC",
            "delta_error_high",
            "rho_base",
            "rho_pen",
        ]
    ].sort_values(["k", "category"])
    slim_path = os.path.join(args.out_dir, "table_mvtec_class_k.csv")
    slim.to_csv(slim_path, index=False, float_format="%.6f")

    visa = df[df["dataset"] == "visa"].copy()
    if len(visa) > 0:
        slim_v = visa[
            [
                "category",
                "k",
                "delta_AUROC",
                "delta_error_high",
                "rho_base",
                "rho_pen",
            ]
        ].sort_values(["k", "category"])
        slim_v_path = os.path.join(args.out_dir, "table_visa_class_k.csv")
        slim_v.to_csv(slim_v_path, index=False, float_format="%.6f")
    else:
        slim_v_path = ""

    # Global summary
    mean_d_auc = float(np.nanmean(df["delta_AUROC"]))
    mean_d_eh = float(np.nanmean(df["delta_error_high"]))
    n_improve_eh = int(np.nansum(df["delta_error_high"] < 0))
    n_improve_auc = int(np.nansum(df["delta_AUROC"] > 0))
    n_tot = len(df)

    summary_lines = [
        "=== Global summary (all analyzed settings) ===",
        f"n_settings: {n_tot}",
        f"mean(delta_AUROC): {mean_d_auc:.6f}",
        f"mean(delta_error_high): {mean_d_eh:.6f}  (negative => lower error in high-I_bin bucket under penalty)",
        f"count(delta_error_high < 0): {n_improve_eh} / {n_tot}",
        f"count(delta_AUROC > 0): {n_improve_auc} / {n_tot}",
        "",
        f"detail: {detail_path}",
        f"mvtec table: {slim_path}",
    ]
    if slim_v_path:
        summary_lines.append(f"visa table: {slim_v_path}")
    elif "visa" in allowed_ds:
        summary_lines.append(
            "visa table: (no visa rows — run inference for visa with DATASETS='mvtec visa' and CHECKPOINT_ROOT containing visa/*.pt)"
        )
    if skipped:
        summary_lines.append(f"\nSkipped / issues: {len(skipped)} (showing up to 15)")
        for s in skipped[:15]:
            summary_lines.append("  " + s)

    summary_text = "\n".join(summary_lines) + "\n"
    print(summary_text)
    with open(os.path.join(args.out_dir, "global_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)


if __name__ == "__main__":
    main()
