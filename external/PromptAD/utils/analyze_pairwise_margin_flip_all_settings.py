"""
Pooled pairwise |m_final| tertiles vs flip rate over all standard CLS runs under result_round1.

Uses the same bucketing as analyze_pairwise_margin_flip_macaroni1_k2.py:
  equal-count tertiles on |m_final| via pd.qcut(..., q=3, labels=["low","mid","high"], duplicates="drop")
applied to all pairs after concatenating every:
  result_round1/*/k_*/pairwise_instability/CLS-*per_sample/pairwise_table.csv

Outputs (default --out_dir):
  - pairwise_margin_flip_tertiles_all81_pooled.csv (full tertile stats)
  - fig_flip_rate_vs_pairwise_margin_all81_pooled_chart.csv (margin_bucket, flip_rate, n_pairs — matches the bar heights)
  - fig_flip_rate_vs_pairwise_margin_all81_pooled.png
  - pairwise_margin_flip_spearman_all81_pooled.txt
  - pairwise_margin_flip_runs_manifest.txt
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def discover_pairwise_csvs(result_root: str) -> list[str]:
    """CLS per-sample pairwise runs only (matches 81-setting convention)."""
    pattern = os.path.join(result_root, "*", "k_*", "pairwise_instability", "CLS-*per_sample", "pairwise_table.csv")
    paths = sorted(glob.glob(pattern))
    out: list[str] = []
    for p in paths:
        if os.path.isfile(p):
            out.append(os.path.abspath(p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_root",
        type=str,
        default="PromptAD/result_round1",
        help="Repo-relative root containing dataset/k_*/pairwise_instability/...",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Default: <result_root>/pairwise_aggregate_81settings/experiments",
    )
    args = parser.parse_args()

    result_root = os.path.abspath(os.path.join(_repo_root(), args.result_root))
    if not os.path.isdir(result_root):
        raise FileNotFoundError(result_root)

    csv_paths = discover_pairwise_csvs(result_root)
    if not csv_paths:
        raise SystemExit(f"No pairwise_table.csv found under {result_root}")

    out_dir = args.out_dir or os.path.join(result_root, "pairwise_aggregate_81settings", "experiments")
    os.makedirs(out_dir, exist_ok=True)

    frames: list[pd.DataFrame] = []
    need = {"m_final", "flip"}
    for p in csv_paths:
        chunk = pd.read_csv(p)
        if not need.issubset(chunk.columns):
            raise ValueError(f"{p}: need columns {need}, got {list(chunk.columns)}")
        frames.append(chunk[list(chunk.columns.intersection(["m_final", "flip", "pair_margin_var"]))])

    df = pd.concat(frames, ignore_index=True)
    m_final = df["m_final"].astype(float)
    flip = df["flip"].astype(float)
    abs_margin = m_final.abs()

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
    tbl_path = os.path.join(out_dir, "pairwise_margin_flip_tertiles_all81_pooled.csv")
    out_tbl.to_csv(tbl_path, index=False)

    chart_tbl = pd.DataFrame(
        {
            "margin_bucket": out_tbl["bucket"].astype(str),
            "flip_rate": out_tbl["flip_rate"].astype(float),
            "n_pairs": out_tbl["count"].astype(int),
        }
    )
    chart_path = os.path.join(out_dir, "fig_flip_rate_vs_pairwise_margin_all81_pooled_chart.csv")
    chart_tbl.to_csv(chart_path, index=False)

    rho = float(abs_margin.corr(flip, method="spearman"))
    spearman_path = os.path.join(out_dir, "pairwise_margin_flip_spearman_all81_pooled.txt")
    with open(spearman_path, "w", encoding="utf-8") as f:
        f.write(f"Spearman(|m_final|, flip) = {rho:.6f}\n")
        f.write(f"n_pairs = {len(df)}\n")
        f.write(f"n_settings = {len(csv_paths)}\n")
        f.write(f"result_root = {result_root}\n")

    manifest_path = os.path.join(out_dir, "pairwise_margin_flip_runs_manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"n_settings = {len(csv_paths)}\n\n")
        for p in csv_paths:
            f.write(os.path.relpath(p, _repo_root()) + "\n")

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    x = np.arange(3)
    rates = out_tbl["flip_rate"].to_numpy(dtype=float)
    ax.bar(x, rates, color="0.45", edgecolor="0.2", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["low margin", "mid margin", "high margin"])
    ax.set_ylabel("Flip rate")
    ax.set_title(f"Flip rate vs pairwise margin (all {len(csv_paths)} CLS settings, pooled)")
    ax.set_ylim(0, max(0.05, float(np.nanmax(rates)) * 1.15))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "fig_flip_rate_vs_pairwise_margin_all81_pooled.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("n_settings:", len(csv_paths))
    print("n_pairs:", len(df))
    print("Wrote:", tbl_path)
    print("Wrote:", chart_path)
    print("Wrote:", fig_path)
    print("Wrote:", spearman_path)
    print("Wrote:", manifest_path)
    print(f"Spearman(|m_final|, flip) = {rho:.6f}")


if __name__ == "__main__":
    main()
