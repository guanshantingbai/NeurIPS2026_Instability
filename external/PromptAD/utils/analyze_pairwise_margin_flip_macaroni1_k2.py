"""
Pairwise |m_final| tertiles vs flip rate for visa-macaroni1-k2 (read pairwise_table.csv only).

Default run: CLS-visa-macaroni1-k2-seed111-per_sample

Outputs (under run experiments/ by default):
  - pairwise_margin_flip_tertiles.csv
  - fig_flip_rate_vs_pairwise_margin_macaroni1_k2.png
  - pairwise_margin_flip_spearman.txt
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairwise_csv",
        type=str,
        default=(
            "PromptAD/result_round1/visa/k_2/pairwise_instability/"
            "CLS-visa-macaroni1-k2-seed111-per_sample/pairwise_table.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Default: <run_dir>/experiments",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(os.path.join(_repo_root(), args.pairwise_csv))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    run_dir = os.path.dirname(csv_path)
    out_dir = args.out_dir or os.path.join(run_dir, "experiments")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    need = {"m_final", "flip"}
    if not need.issubset(df.columns):
        raise ValueError(f"Need columns {need}, got {list(df.columns)}")

    m_final = df["m_final"].astype(float)
    flip = df["flip"].astype(float)
    abs_margin = m_final.abs()

    # Equal-count tertiles on |m_final|
    try:
        bucket = pd.qcut(abs_margin, q=3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError as e:
        raise SystemExit(f"qcut failed (need distinct margins): {e}") from e

    df = df.assign(abs_margin=abs_margin, margin_bucket=bucket)

    rows = []
    for name in ["low", "mid", "high"]:
        sub = df[df["margin_bucket"] == name]
        lo = float(sub["abs_margin"].min()) if len(sub) else float("nan")
        hi = float(sub["abs_margin"].max()) if len(sub) else float("nan")
        mean_abs = float(sub["abs_margin"].mean()) if len(sub) else float("nan")
        mean_pm_var = (
            float(sub["pair_margin_var"].mean()) if len(sub) and "pair_margin_var" in sub.columns else float("nan")
        )
        rows.append(
            {
                "bucket": name,
                "count": int(len(sub)),
                "flip_rate": float(sub["flip"].mean()) if len(sub) else float("nan"),
                "abs_m_final_min": lo,
                "abs_m_final_max": hi,
                "abs_m_final_mean": mean_abs,
                "mean_pair_margin_var": mean_pm_var,
            }
        )

    out_tbl = pd.DataFrame(rows)
    tbl_path = os.path.join(out_dir, "pairwise_margin_flip_tertiles.csv")
    out_tbl.to_csv(tbl_path, index=False)

    # Spearman(|m_final|, flip) — use absolute margin as specified
    rho = float(abs_margin.corr(flip, method="spearman"))
    spearman_path = os.path.join(out_dir, "pairwise_margin_flip_spearman.txt")
    with open(spearman_path, "w", encoding="utf-8") as f:
        f.write(f"Spearman(|m_final|, flip) = {rho:.6f}\n")
        f.write(f"n_pairs = {len(df)}\n")

    # Bar chart
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    x = np.arange(3)
    rates = out_tbl["flip_rate"].to_numpy(dtype=float)
    ax.bar(x, rates, color="0.45", edgecolor="0.2", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["low margin", "mid margin", "high margin"])
    ax.set_ylabel("Flip rate")
    ax.set_title("Flip rate vs pairwise margin (visa-macaroni1-k2)")
    ax.set_ylim(0, max(0.05, float(np.nanmax(rates)) * 1.15))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "fig_flip_rate_vs_pairwise_margin_macaroni1_k2.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", tbl_path)
    print("Wrote:", fig_path)
    print("Wrote:", spearman_path)
    print(f"Spearman(|m_final|, flip) = {rho:.6f}")


if __name__ == "__main__":
    main()
