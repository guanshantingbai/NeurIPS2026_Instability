"""
Figure 3.1.1: histogram of pairwise flip_rate_mean across all PromptAD settings.
Reads exp_summary_all.csv (or equivalent with flip_rate_mean column).
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=os.path.join("PromptAD", "result_round1", "exp_summary_all.csv"),
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default=(
            "PromptAD/result_round1/visa/k_2/pairwise_instability/"
            "CLS-visa-macaroni1-k2-seed111-per_sample/experiments/"
            "instability_fliprate_distribution.png"
        ),
    )
    parser.add_argument(
        "--out_pdf",
        type=str,
        default=(
            "PromptAD/result_round1/visa/k_2/pairwise_instability/"
            "CLS-visa-macaroni1-k2-seed111-per_sample/experiments/"
            "instability_fliprate_distribution.pdf"
        ),
    )
    parser.add_argument(
        "--out_stats",
        type=str,
        default=(
            "PromptAD/result_round1/visa/k_2/pairwise_instability/"
            "CLS-visa-macaroni1-k2-seed111-per_sample/experiments/"
            "instability_fliprate_distribution_stats.txt"
        ),
    )
    args = parser.parse_args()

    csv_path = os.path.normpath(os.path.abspath(args.summary_csv))
    df = pd.read_csv(csv_path)
    if "flip_rate_mean" not in df.columns:
        raise ValueError(f"Column flip_rate_mean missing in {csv_path}")

    x = pd.to_numeric(df["flip_rate_mean"], errors="coerce").dropna().to_numpy()
    n = len(x)
    mean_v = float(np.mean(x))
    med_v = float(np.median(x))
    min_v = float(np.min(x))
    max_v = float(np.max(x))

    stats_lines = [
        f"number_of_settings: {n}",
        f"mean_flip_rate: {mean_v:.6f}",
        f"median_flip_rate: {med_v:.6f}",
        f"min_flip_rate: {min_v:.6f}",
        f"max_flip_rate: {max_v:.6f}",
        f"source_csv: {csv_path}",
    ]
    out_stats = os.path.normpath(os.path.abspath(args.out_stats))
    os.makedirs(os.path.dirname(out_stats), exist_ok=True)
    with open(out_stats, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines) + "\n")

    fs = 12
    plt.rcParams.update(
        {
            "font.size": fs,
            "axes.labelsize": fs,
            "axes.titlesize": fs,
            "xtick.labelsize": fs,
            "ytick.labelsize": fs,
            "legend.fontsize": fs,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(5.0, 3.2), layout="constrained")

    bins = 15
    xmax = min(1.0, max_v + 0.06)
    counts, _, _ = ax.hist(
        x,
        bins=bins,
        range=(0.0, xmax),
        color="0.45",
        edgecolor="0.15",
        linewidth=0.6,
    )

    ax.axvline(
        mean_v,
        color="red",
        linestyle="-",
        linewidth=1.2,
        label=f"mean flip rate = {mean_v:.2f}",
    )

    ax.set_xlabel("Pairwise flip rate")
    ax.set_ylabel("Number of settings")

    ax.set_xlim(0, xmax)
    ymax = float(np.max(counts)) if len(counts) else 1.0
    ax.set_ylim(0, ymax * 1.12)

    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.5")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_png = os.path.normpath(os.path.abspath(args.out_png))
    out_pdf = os.path.normpath(os.path.abspath(args.out_pdf))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print("Wrote:", out_png)
    print("Wrote:", out_pdf)
    print("Wrote:", out_stats)


if __name__ == "__main__":
    main()
