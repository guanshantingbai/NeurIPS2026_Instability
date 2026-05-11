#!/usr/bin/env python3
"""
Aggregate statistics after a full test_cls run (Strategy 1: thresholded instability penalty).

Reads:
  - {infer_root}/{dataset}/k_{k}/csv/Seed_*-results.csv  -> per-class AUROC (i_roc vs i_roc_baseline)
  - {infer_root}/**/CLS-*-per_sample_instability.csv     -> u_abs, high_u_mask, ranking errors

Writes:
  - summary_auroc_by_group.csv
  - summary_instability_by_setting.csv
  - summary_global.txt

Run from PromptAD repo root:
  python utils/summarize_thresholded_infer_run.py --infer-root result_offline_infer
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _parse_results_path(path: str) -> Optional[Tuple[str, int, int]]:
    """infer_root/mvtec/k_1/csv/Seed_111-results.csv -> (mvtec, 1, 111)"""
    norm = os.path.normpath(path)
    m = re.search(r"/([^/]+)/k_(\d+)/csv/Seed_(\d+)-results\.csv$", norm)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def _parse_instability_path(path: str) -> Optional[Tuple[str, str, int, int]]:
    """
    CLS-{dataset}-{class}-k{k}-seed{seed}-per_sample_instability.csv
    """
    base = os.path.basename(path)
    m = re.match(
        r"^CLS-(mvtec|visa)-(.+)-k(\d+)-seed(\d+)-per_sample_instability\.csv$",
        base,
    )
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3)), int(m.group(4))


def load_all_auroc(infer_root: str) -> pd.DataFrame:
    pattern = os.path.join(infer_root, "*", "k_*", "csv", "Seed_*-results.csv")
    rows: List[dict] = []
    for path in sorted(glob.glob(pattern)):
        parsed = _parse_results_path(path)
        if parsed is None:
            continue
        dataset, k_shot, seed = parsed
        df = pd.read_csv(path, index_col=0)
        if "i_roc" not in df.columns or "i_roc_baseline" not in df.columns:
            continue
        for cls, r in df.iterrows():
            ib = float(r["i_roc"])
            ibase = float(r["i_roc_baseline"])
            rows.append(
                {
                    "dataset": dataset,
                    "k_shot": k_shot,
                    "seed": seed,
                    "class": str(cls),
                    "i_roc_thresholded": ib,
                    "i_roc_baseline": ibase,
                    "delta_auroc": ib - ibase,
                }
            )
    return pd.DataFrame(rows)


def summarize_one_instability_csv(
    path: str,
    ranking_err_pen_col: str = "ranking_error_0p1",
) -> Optional[dict]:
    parsed = _parse_instability_path(path)
    if parsed is None:
        return None
    dataset, category, k_shot, seed = parsed
    df = pd.read_csv(path)
    need = {"u_abs", "high_u_mask", "ranking_error_baseline", ranking_err_pen_col}
    if not need.issubset(df.columns):
        return None

    u = pd.to_numeric(df["u_abs"], errors="coerce").to_numpy(dtype=float)
    mask = pd.to_numeric(df["high_u_mask"], errors="coerce").fillna(0).to_numpy(dtype=int) > 0
    e0 = pd.to_numeric(df["ranking_error_baseline"], errors="coerce").to_numpy(dtype=float)
    e1 = pd.to_numeric(df[ranking_err_pen_col], errors="coerce").to_numpy(dtype=float)
    n = len(df)
    if n == 0:
        return None

    fin = np.isfinite(e0) & np.isfinite(e1)
    rho = np.nan
    if fin.sum() >= 3 and np.nanstd(u[fin]) > 0 and np.nanstd(e0[fin]) > 0:
        r, _ = spearmanr(u[fin], e0[fin])
        rho = float(r) if r == r else np.nan

    penalized_frac = float(mask.mean()) if n else np.nan
    mean_u = float(np.nanmean(u))
    mean_err_base = float(np.nanmean(e0[fin]))
    mean_err_pen = float(np.nanmean(e1[fin]))
    delta_err_all = mean_err_pen - mean_err_base

    if mask.any() and np.isfinite(e0[mask & fin]).any():
        hb = float(np.nanmean(e0[mask & fin]))
        hp = float(np.nanmean(e1[mask & fin]))
        delta_err_high_u = hp - hb
    else:
        hb = hp = delta_err_high_u = np.nan

    return {
        "dataset": dataset,
        "category": category,
        "k_shot": k_shot,
        "seed": seed,
        "n_samples": n,
        "penalized_frac": penalized_frac,
        "mean_u_abs": mean_u,
        "spearman_u_abs_err_base": rho,
        "mean_rank_err_baseline": mean_err_base,
        "mean_rank_err_penalized": mean_err_pen,
        "delta_mean_rank_err_all": delta_err_all,
        "mean_rank_err_high_u_baseline": hb,
        "mean_rank_err_high_u_penalized": hp,
        "delta_mean_rank_err_high_u": delta_err_high_u,
        "path": path,
    }


def load_all_instability(infer_root: str, ranking_err_pen_col: str) -> pd.DataFrame:
    pattern = os.path.join(infer_root, "**", "CLS-*-per_sample_instability.csv")
    rows: List[dict] = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        row = summarize_one_instability_csv(path, ranking_err_pen_col=ranking_err_pen_col)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer-root",
        type=str,
        default="result_offline_infer",
        help="Root written by test_cls / run_cls_infer_parallel2.sh",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: <infer-root>/thresholded_summary)",
    )
    parser.add_argument(
        "--ranking-error-pen-col",
        type=str,
        default="ranking_error_0p1",
        help="Column for penalized ranking error (match --instability-penalty-lambda 0.1)",
    )
    args = parser.parse_args()
    infer_root = os.path.abspath(args.infer_root)
    if not os.path.isdir(infer_root):
        print(f"[ERROR] infer-root not a directory: {infer_root}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir.strip()
    if not out_dir:
        out_dir = os.path.join(infer_root, "thresholded_summary")
    os.makedirs(out_dir, exist_ok=True)

    auroc_df = load_all_auroc(infer_root)
    inst_df = load_all_instability(infer_root, args.ranking_error_pen_col)

    lines: List[str] = []
    lines.append(f"infer_root: {infer_root}")
    lines.append("")

    # ----- AUROC -----
    lines.append("=== 1) AUROC (thresholded vs harmonic baseline) ===")
    if auroc_df.empty:
        lines.append("No Seed_*-results.csv found.")
    else:
        auroc_path = os.path.join(out_dir, "summary_auroc_long.csv")
        auroc_df.to_csv(auroc_path, index=False, float_format="%.6f")

        g = auroc_df.groupby(["dataset", "k_shot"], as_index=False).agg(
            n_classes=("class", "count"),
            mean_i_thr=("i_roc_thresholded", "mean"),
            mean_i_base=("i_roc_baseline", "mean"),
            mean_delta=("delta_auroc", "mean"),
            median_delta=("delta_auroc", "median"),
        )
        g_path = os.path.join(out_dir, "summary_auroc_by_dataset_k.csv")
        g.to_csv(g_path, index=False, float_format="%.6f")

        n_set = len(auroc_df)
        n_up = int((auroc_df["delta_auroc"] > 0).sum())
        n_down = int((auroc_df["delta_auroc"] < 0).sum())
        n_eq = int((auroc_df["delta_auroc"] == 0).sum())

        lines.append(f"Per-class rows: {n_set}")
        lines.append(
            f"Delta AUROC (thresholded - baseline): "
            f"mean={auroc_df['delta_auroc'].mean():.4f}, "
            f"median={auroc_df['delta_auroc'].median():.4f}, "
            f"std={auroc_df['delta_auroc'].std():.4f}"
        )
        lines.append(f"Count delta > 0 / == 0 / < 0: {n_up} / {n_eq} / {n_down}")
        lines.append("")
        lines.append("Mean AUROC by (dataset, k_shot):")
        for _, r in g.iterrows():
            lines.append(
                f"  {r['dataset']} k={int(r['k_shot'])}: "
                f"mean_thr={r['mean_i_thr']:.2f}, mean_base={r['mean_i_base']:.2f}, "
                f"mean_delta={r['mean_delta']:.4f}  (n={int(r['n_classes'])})"
            )
        lines.append(f"[saved] {auroc_path}")
        lines.append(f"[saved] {g_path}")

    lines.append("")

    # ----- Instability-related -----
    lines.append("=== 2) Instability-related (per-class CSVs, λ=0.1 column) ===")
    if inst_df.empty:
        lines.append("No CLS-*-per_sample_instability.csv found.")
    else:
        inst_path = os.path.join(out_dir, "summary_instability_by_setting.csv")
        inst_df.drop(columns=["path"], errors="ignore").to_csv(
            inst_path, index=False, float_format="%.6f"
        )

        w = inst_df["n_samples"].to_numpy(dtype=float)
        wsum = w.sum()

        def wmean(col: str) -> float:
            x = inst_df[col].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(w) & (w > 0)
            if not m.any():
                return float("nan")
            return float(np.sum(x[m] * w[m]) / np.sum(w[m]))

        lines.append(f"Settings (class × k × seed files): {len(inst_df)}")
        lines.append(
            f"Total test images (sum of n_samples): {int(wsum)}"
        )
        lines.append(
            f"Penalized fraction (high_u_mask mean), unweighted across settings: "
            f"mean={inst_df['penalized_frac'].mean():.4f}, "
            f"median={inst_df['penalized_frac'].median():.4f}"
        )
        lines.append(
            f"Penalized fraction, weighted by n_samples: {wmean('penalized_frac'):.4f}"
        )
        lines.append(
            f"mean |s_sem - s_vis| (u_abs), weighted by n_samples: {wmean('mean_u_abs'):.6f}"
        )
        lines.append(
            f"Spearman(u_abs, ranking_error_baseline), mean across settings: "
            f"{inst_df['spearman_u_abs_err_base'].mean():.4f}"
        )
        lines.append(
            f"Delta mean ranking error (all samples, pen - base), "
            f"mean across settings: {inst_df['delta_mean_rank_err_all'].mean():.6f}"
        )
        lines.append(
            f"Delta mean ranking error on {{u > tau}} only, mean across settings: "
            f"{inst_df['delta_mean_rank_err_high_u'].mean():.6f}"
        )
        lines.append(f"[saved] {inst_path}")

    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(out_dir, "summary_global.txt"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    os.chdir(_REPO_ROOT)
    main()
