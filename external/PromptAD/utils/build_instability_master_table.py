"""
Scan all pairwise_instability runs under result_round1 and build one master CSV
with exp5 correlations, tertile bucket means, rejection errors, and monotonicity.

Does not read pairwise_table.csv — only exp5_sample_ranking_error.csv and
exp5_instability_rejection.csv per run.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _parse_k_from_dirname(k_dirname: str) -> Optional[int]:
    m = re.fullmatch(r"k_(\d+)", k_dirname)
    return int(m.group(1)) if m else None


def _category_from_run_name(run_name: str, dataset: str) -> str:
    """Strip CLS-{dataset}- prefix and trailing -k{digits}... to get category slug."""
    prefix = f"CLS-{dataset}-"
    if not run_name.startswith(prefix):
        return ""
    rest = run_name[len(prefix) :]
    parts = re.split(r"-k\d+", rest, maxsplit=1)
    return parts[0] if parts else ""


def _tertile_mean_errors(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Match exp6: rank(I_bin) then pd.qcut(..., q=3, labels low/mid/high)."""
    x = df["I_bin"].astype(float)
    y = df["error"].astype(float)
    ranks = x.rank(method="first")
    try:
        bucket = pd.qcut(ranks, q=3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        return (float("nan"), float("nan"), float("nan"))

    tmp = pd.DataFrame({"_bucket": bucket, "error": y})
    agg = tmp.groupby("_bucket", observed=True)["error"].mean()
    low = float(agg["low"]) if "low" in agg.index else float("nan")
    mid = float(agg["mid"]) if "mid" in agg.index else float("nan")
    high = float(agg["high"]) if "high" in agg.index else float("nan")
    return low, mid, high


def _correlations_ibin_error(df: pd.DataFrame) -> Tuple[float, float]:
    sub = df[["I_bin", "error"]].dropna()
    if len(sub) < 2:
        return float("nan"), float("nan")
    ib = sub["I_bin"].to_numpy(dtype=float)
    er = sub["error"].to_numpy(dtype=float)
    if np.std(ib) == 0 or np.std(er) == 0:
        return float("nan"), float("nan")
    p = float(pearsonr(ib, er)[0])
    s = float(spearmanr(ib, er)[0])
    return p, s


def _read_rejection_metrics(rejection_csv: str) -> Optional[Dict[str, Any]]:
    """baseline_error, error@10/20/30 from exp5_instability_rejection.csv."""
    try:
        rej = pd.read_csv(rejection_csv, comment="#")
    except Exception as e:
        warnings.warn(f"Failed to read {rejection_csv}: {e}")
        return None

    need = {"setting", "mean_error_all"}
    if not need.issubset(rej.columns):
        warnings.warn(f"Missing columns in {rejection_csv}: {need - set(rej.columns)}")
        return None

    mapping = {
        "baseline": "baseline_error",
        "reject_10": "error@10",
        "reject_20": "error@20",
        "reject_30": "error@30",
    }
    out: Dict[str, Any] = {}
    for setting, col_key in mapping.items():
        row = rej.loc[rej["setting"] == setting, "mean_error_all"]
        if row.empty:
            out[col_key] = float("nan")
        else:
            out[col_key] = float(row.iloc[0])

    b = out["baseline_error"]
    for pct, err_key in [(10, "error@10"), (20, "error@20"), (30, "error@30")]:
        e = out[err_key]
        rel_key = f"rel_drop@{pct}"
        if b == b and b != 0 and e == e:
            out[rel_key] = (b - e) / b
        else:
            out[rel_key] = float("nan")

    e0 = out["baseline_error"]
    e10, e20, e30 = out["error@10"], out["error@20"], out["error@30"]
    vals = [e0, e10, e20, e30]
    if all(v == v for v in vals):
        out["is_monotonic"] = bool(e0 > e10 > e20 > e30)
    else:
        out["is_monotonic"] = False

    return out


def _parse_run_from_sample_csv(sample_csv: str) -> Optional[Dict[str, Any]]:
    """
    Expect: .../{dataset}/k_{k}/pairwise_instability/{run_name}/experiments/exp5_sample_ranking_error.csv
    """
    sample_csv = os.path.normpath(sample_csv)
    exp_dir = os.path.dirname(sample_csv)
    run_dir = os.path.dirname(exp_dir)
    pairwise_dir = os.path.dirname(run_dir)
    k_dir = os.path.dirname(pairwise_dir)
    dataset_dir = os.path.dirname(k_dir)

    if os.path.basename(pairwise_dir) != "pairwise_instability":
        return None
    if os.path.basename(exp_dir) != "experiments":
        return None

    run_name = os.path.basename(run_dir)
    dataset = os.path.basename(dataset_dir)
    k_name = os.path.basename(k_dir)
    k = _parse_k_from_dirname(k_name)
    if k is None:
        return None

    category = _category_from_run_name(run_name, dataset)
    return {
        "dataset": dataset,
        "category": category,
        "k": k,
        "run_name": run_name,
        "experiments_dir": exp_dir,
    }


def _process_run(sample_csv: str) -> Optional[Dict[str, Any]]:
    meta = _parse_run_from_sample_csv(sample_csv)
    if meta is None:
        warnings.warn(f"Path layout not recognized, skip: {sample_csv}")
        return None

    rej_path = os.path.join(meta["experiments_dir"], "exp5_instability_rejection.csv")
    if not os.path.isfile(rej_path):
        warnings.warn(f"Missing exp5_instability_rejection.csv for {sample_csv}")
        return None

    try:
        df = pd.read_csv(sample_csv)
    except Exception as e:
        warnings.warn(f"Failed to read {sample_csv}: {e}")
        return None

    need = {"I_bin", "error"}
    if not need.issubset(df.columns):
        warnings.warn(f"Missing I_bin/error in {sample_csv}")
        return None

    pearson_r, spearman_r = _correlations_ibin_error(df)
    low_e, mid_e, high_e = _tertile_mean_errors(df)
    rej = _read_rejection_metrics(rej_path)
    if rej is None:
        return None

    row: Dict[str, Any] = {
        "dataset": meta["dataset"],
        "category": meta["category"],
        "k": meta["k"],
        "run_name": meta["run_name"],
        "spearman_ibin_error": spearman_r,
        "pearson_ibin_error": pearson_r,
        "mean_error_low_I_bin": low_e,
        "mean_error_mid_I_bin": mid_e,
        "mean_error_high_I_bin": high_e,
        "baseline_error": rej["baseline_error"],
        "error@10": rej["error@10"],
        "error@20": rej["error@20"],
        "error@30": rej["error@30"],
        "rel_drop@10": rej["rel_drop@10"],
        "rel_drop@20": rej["rel_drop@20"],
        "rel_drop@30": rej["rel_drop@30"],
        "is_monotonic": rej["is_monotonic"],
    }
    return row


def _glob_sample_csvs(result_root: str) -> List[str]:
    pattern = os.path.join(
        result_root,
        "*",
        "k_*",
        "pairwise_instability",
        "*",
        "experiments",
        "exp5_sample_ranking_error.csv",
    )
    return sorted(glob.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build instability master table from all exp5 runs")
    parser.add_argument(
        "--result_root",
        type=str,
        default=os.path.join("PromptAD", "result_round1"),
        help="Root directory (contains dataset subdirs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("PromptAD", "result_round1", "instability_master_table.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    result_root = os.path.normpath(os.path.abspath(args.result_root))
    out_path = os.path.normpath(os.path.abspath(args.output))

    if not os.path.isdir(result_root):
        raise FileNotFoundError(f"result_root not found: {result_root}")

    sample_files = _glob_sample_csvs(result_root)
    if not sample_files:
        print(f"No exp5_sample_ranking_error.csv found under {result_root}")
        rows: List[Dict[str, Any]] = []
    else:
        rows = []
        for p in sample_files:
            r = _process_run(p)
            if r is not None:
                rows.append(r)

    column_order = [
        "dataset",
        "category",
        "k",
        "run_name",
        "spearman_ibin_error",
        "pearson_ibin_error",
        "mean_error_low_I_bin",
        "mean_error_mid_I_bin",
        "mean_error_high_I_bin",
        "baseline_error",
        "error@10",
        "error@20",
        "error@30",
        "rel_drop@10",
        "rel_drop@20",
        "rel_drop@30",
        "is_monotonic",
    ]
    out_df = pd.DataFrame(rows)
    if len(out_df):
        sort_cols = [c for c in ["dataset", "k", "category", "run_name"] if c in out_df.columns]
        if sort_cols:
            out_df = out_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        existing = [c for c in column_order if c in out_df.columns]
        rest = [c for c in out_df.columns if c not in existing]
        out_df = out_df[existing + rest]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
