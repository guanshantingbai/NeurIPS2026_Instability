"""
Post-hoc analyses for exp5 sample-level ranking error and rejection curves.

Tasks:
  1) Extreme samples (high I_bin + high error) -> exp6_extreme_samples.csv
  2) I_bin vs error scatter, correlations, tertile buckets -> PNG + CSV
  3) Rejection rate vs mean_error_all curve -> PNG

Outputs go to the same directory as the input exp5_sample_ranking_error.csv (typically .../experiments/).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paper-ready exp6 / companion figures: axis text ~2× default (~10pt)
_EXP6_AXIS_FONTSIZE = 20
_EXP6_POINT_ANNOT_FONTSIZE = 18
_EXP6_IBIN_ERROR_TREND_COLOR = "#D98880"


def _experiments_dir(sample_error_csv: str) -> str:
    d = os.path.dirname(os.path.abspath(sample_error_csv))
    return d


def task1_extreme_samples(df: pd.DataFrame, out_dir: str) -> str:
    """Filter I_bin > 0.3 and error > 0.5; save CSV and print counts."""
    need = {"image_path", "image_label", "I_bin", "error", "harmonic_score"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns: {need - set(df.columns)}")

    sub = df[(df["I_bin"] > 0.3) & (df["error"] > 0.5)].copy()
    out_path = os.path.join(out_dir, "exp6_extreme_samples.csv")
    sub[["image_path", "image_label", "I_bin", "error", "harmonic_score"]].to_csv(
        out_path, index=False
    )

    n = len(sub)
    n_a = int((sub["image_label"] == "anomaly").sum())
    n_n = int((sub["image_label"] == "normal").sum())
    print("\n=== Task 1: extreme samples (I_bin>0.3, error>0.5) ===")
    print(f"Selected count: {n}")
    print(f"  anomaly: {n_a}, normal: {n_n}")
    print(f"Wrote: {out_path}")
    return out_path


def task2_ibin_error_distribution(df: pd.DataFrame, out_dir: str) -> Tuple[str, str]:
    """Scatter I_bin vs error, correlations, tertile bucket stats."""
    need = {"I_bin", "error"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns: {need - set(df.columns)}")

    x = df["I_bin"].astype(float)
    y = df["error"].astype(float)

    pearson = float(x.corr(y, method="pearson"))
    spearman = float(x.corr(y, method="spearman"))
    print("\n=== Task 2: I_bin vs error ===")
    print(f"Pearson correlation:  {pearson:.6f}")
    print(f"Spearman correlation: {spearman:.6f}")

    # Scatter (alpha for de-cluttering) + linear trend for reviewers
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.3, s=7, edgecolors="none", rasterized=True)

    mask = x.notna() & y.notna()
    xx = x[mask].to_numpy(dtype=float)
    yy = y[mask].to_numpy(dtype=float)
    if len(xx) >= 2 and float(np.std(xx)) > 1e-12:
        coef = np.polyfit(xx, yy, 1)
        xs = np.linspace(float(np.min(xx)), float(np.max(xx)), 100)
        ax.plot(
            xs,
            np.polyval(coef, xs),
            color=_EXP6_IBIN_ERROR_TREND_COLOR,
            linewidth=2.0,
            zorder=3,
        )

    if spearman == spearman:
        ax.text(
            0.98,
            0.98,
            rf"Spearman $\rho$ = {spearman:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=_EXP6_AXIS_FONTSIZE,
            color=_EXP6_IBIN_ERROR_TREND_COLOR,
        )

    ax.set_xlabel("I_bin", fontsize=_EXP6_AXIS_FONTSIZE)
    ax.set_ylabel("error (ranking)", fontsize=_EXP6_AXIS_FONTSIZE)
    ax.tick_params(axis="both", labelsize=_EXP6_AXIS_FONTSIZE)
    fig.tight_layout()
    scatter_path = os.path.join(out_dir, "exp6_ibin_vs_error_scatter.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {scatter_path}")

    # Tertile buckets on I_bin (by rank / qcut for ~equal counts)
    ranks = x.rank(method="first")
    try:
        bucket = pd.qcut(ranks, q=3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        bucket = pd.qcut(ranks, q=min(3, len(ranks.unique())), duplicates="drop")

    tmp = df.assign(_bucket=bucket)
    bucket_stats = (
        tmp.groupby("_bucket", observed=True)
        .agg(sample_count=("error", "size"), mean_error=("error", "mean"))
        .reset_index()
        .rename(columns={"_bucket": "I_bin_bucket"})
    )
    bucket_path = os.path.join(out_dir, "exp6_ibin_error_bucket.csv")
    bucket_stats.to_csv(bucket_path, index=False)
    print(f"Wrote: {bucket_path}")
    return scatter_path, bucket_path


def _setting_to_reject_pct(setting: str) -> float:
    if setting == "baseline":
        return 0.0
    if setting == "reject_10":
        return 10.0
    if setting == "reject_20":
        return 20.0
    if setting == "reject_30":
        return 30.0
    return float("nan")


def task3_rejection_curve(rejection_csv: str, out_dir: str) -> str:
    """Line plot: rejection rate (%) vs mean_error_all."""
    rej = pd.read_csv(rejection_csv, comment="#")
    if "setting" not in rej.columns or "mean_error_all" not in rej.columns:
        raise ValueError("rejection CSV must contain setting and mean_error_all")

    rej = rej.copy()
    rej["reject_pct"] = rej["setting"].map(_setting_to_reject_pct)
    rej = rej.dropna(subset=["reject_pct", "mean_error_all"]).sort_values("reject_pct")

    xs = rej["reject_pct"].to_numpy()
    ys = rej["mean_error_all"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, ys, marker="o", linewidth=1.5)
    ax.set_xlabel("rejection rate (%)", fontsize=_EXP6_AXIS_FONTSIZE)
    ax.set_ylabel("Mean ranking error", fontsize=_EXP6_AXIS_FONTSIZE)
    ax.tick_params(axis="both", labelsize=_EXP6_AXIS_FONTSIZE)
    ax.set_xticks(xs)
    for px, py in zip(xs, ys):
        ax.annotate(
            f"{py:.4f}",
            (px, py),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=_EXP6_POINT_ANNOT_FONTSIZE,
        )
    fig.tight_layout()
    out_path = os.path.join(out_dir, "exp6_rejection_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("\n=== Task 3: rejection curve ===")
    print(rej[["setting", "reject_pct", "mean_error_all"]].to_string(index=False))
    print(f"Wrote: {out_path}")
    return out_path


def run_exp6_all(experiments_dir: str, *, quiet: bool = False) -> None:
    """
    Write exp6 CSV/PNGs into experiments_dir using exp5_sample_ranking_error.csv and
    exp5_instability_rejection.csv in the same folder.
    """
    experiments_dir = os.path.normpath(experiments_dir)
    sample_path = os.path.join(experiments_dir, "exp5_sample_ranking_error.csv")
    rejection_path = os.path.join(experiments_dir, "exp5_instability_rejection.csv")
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(sample_path)
    if not os.path.isfile(rejection_path):
        raise FileNotFoundError(rejection_path)
    os.makedirs(experiments_dir, exist_ok=True)
    df = pd.read_csv(sample_path)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            task1_extreme_samples(df, experiments_dir)
            task2_ibin_error_distribution(df, experiments_dir)
            task3_rejection_curve(rejection_path, experiments_dir)
    else:
        task1_extreme_samples(df, experiments_dir)
        task2_ibin_error_distribution(df, experiments_dir)
        task3_rejection_curve(rejection_path, experiments_dir)


def main() -> None:
    default_sample = (
        "PromptAD/result_round1/visa/k_2/pairwise_instability/"
        "CLS-visa-candle-k2-seed111-per_sample/experiments/exp5_sample_ranking_error.csv"
    )
    default_rejection = (
        "PromptAD/result_round1/visa/k_2/pairwise_instability/"
        "CLS-visa-candle-k2-seed111-per_sample/experiments/exp5_instability_rejection.csv"
    )

    parser = argparse.ArgumentParser(description="Analyze instability behavior (exp6 outputs)")
    parser.add_argument(
        "--exp5_sample_csv",
        type=str,
        default=default_sample,
        help="Path to exp5_sample_ranking_error.csv",
    )
    parser.add_argument(
        "--exp5_rejection_csv",
        type=str,
        default=default_rejection,
        help="Path to exp5_instability_rejection.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory (default: same dir as exp5_sample_csv)",
    )
    args = parser.parse_args()

    sample_path = os.path.normpath(args.exp5_sample_csv)
    rejection_path = os.path.normpath(args.exp5_rejection_csv)
    out_dir = args.output_dir or _experiments_dir(sample_path)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(sample_path):
        raise FileNotFoundError(sample_path)
    if not os.path.isfile(rejection_path):
        raise FileNotFoundError(rejection_path)

    df = pd.read_csv(sample_path)
    print(f"Loaded {len(df)} rows from {sample_path}")
    print(f"Output directory: {out_dir}")

    run_exp6_all(out_dir, quiet=False)

    print("\nDone. All exp6 artifacts written to:", out_dir)


if __name__ == "__main__":
    main()
