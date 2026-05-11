"""
Minimal Appendix A.1 (PromptAD): select from existing result_round1 CSV/JSON only.

Excludes main-paper setting visa-macaroni1-k2 from per-setting exports.
Does not retrain or recompute model scores.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGSIZE = (6.5, 4.5)
DPI = 300

# Main text setting visa-macaroni1-k2 is not duplicated under extra_category/ or different_k/.

# Priority: capsules, pcb2, candle (visa, k=2)
EXTRA_CATEGORY_CANDIDATES = [
    ("visa", "capsules", 2),
    ("visa", "pcb2", 2),
    ("visa", "candle", 2),
]

# Different k for macaroni1
DIFF_K_CANDIDATES = [
    ("visa", "macaroni1", 4),
    ("visa", "macaroni1", 8),
]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run_dir(result_root: str, dataset: str, k: int, run_name: str) -> str:
    return os.path.join(
        result_root, dataset, f"k_{k}", "pairwise_instability", run_name
    )


def default_run_name(dataset: str, category: str, k: int) -> str:
    return f"CLS-{dataset}-{category}-k{k}-seed111-per_sample"


def find_existing_run(
    result_root: str, dataset: str, category: str, k: int
) -> Optional[str]:
    rn = default_run_name(dataset, category, k)
    rd = run_dir(result_root, dataset, k, rn)
    summary = os.path.join(rd, "summary.json")
    if os.path.isfile(summary):
        return rn
    return None


def pick_extra_category(result_root: str) -> Tuple[str, str, int, str]:
    for dataset, category, k in EXTRA_CATEGORY_CANDIDATES:
        rn = find_existing_run(result_root, dataset, category, k)
        if rn:
            return dataset, category, k, rn
    raise FileNotFoundError("No extra-category run among capsules/pcb2/candle (k=2)")


def pick_diff_k(result_root: str) -> Tuple[str, str, int, str]:
    for dataset, category, k in DIFF_K_CANDIDATES:
        rn = find_existing_run(result_root, dataset, category, k)
        if rn:
            return dataset, category, k, rn
    raise FileNotFoundError("No visa-macaroni1 k4 or k8 run found")


def iter_all_exp5_runs(result_root: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(
        result_root, "*", "k_*", "pairwise_instability", "*", "experiments", "exp5_sample_ranking_error.csv"
    )
    rows = []
    for p in sorted(glob.glob(pattern)):
        run_dir_path = os.path.dirname(os.path.dirname(p))
        rel = os.path.relpath(run_dir_path, result_root)
        parts = rel.split(os.sep)
        if len(parts) < 4:
            continue
        dataset, kfolder, _, run_name = parts[0], parts[1], parts[2], parts[3]
        k = int(kfolder.split("_", 1)[1])
        cat = run_name
        if run_name.startswith(f"CLS-{dataset}-"):
            suf = f"-k{k}-"
            pref = f"CLS-{dataset}-"
            if suf in run_name:
                cat = run_name[len(pref) : run_name.index(suf)]
        rows.append(
            {
                "path": p,
                "dataset": dataset,
                "category": cat,
                "k": k,
                "run_name": run_name,
            }
        )
    return rows


def spearman_pearson(df: pd.DataFrame) -> Tuple[float, float, int]:
    if "I_bin" not in df.columns or "error" not in df.columns:
        return float("nan"), float("nan"), 0
    x = df["I_bin"].astype(float)
    y = df["error"].astype(float)
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    xx = x[m]
    yy = y[m]
    if float(np.nanstd(xx.to_numpy())) < 1e-12 or float(np.nanstd(yy.to_numpy())) < 1e-12:
        return float("nan"), float("nan"), n
    return float(xx.corr(yy, method="spearman")), float(xx.corr(yy, method="pearson")), n


def rejection_series(path: str) -> Optional[np.ndarray]:
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, comment="#")
    order = ["baseline", "reject_10", "reject_20", "reject_30"]
    ys = []
    for s in order:
        row = df.loc[df["setting"] == s, "mean_error_all"]
        if row.empty:
            return None
        ys.append(float(row.iloc[0]))
    return np.array(ys, dtype=float)


def plot_spearman_histogram(spearmans: List[float], out_path: str) -> None:
    vals = np.array([s for s in spearmans if s == s], dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if len(vals) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(vals, bins=min(20, max(8, len(vals) // 4)), color="C0", edgecolor="0.4", linewidth=0.4)
    ax.set_xlabel(r"Spearman $\rho$ ($I_{\mathrm{bin}}$, error)")
    ax.set_ylabel("Number of runs")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mean_rejection(all_curves: List[np.ndarray], out_path: str) -> None:
    xs = np.array([0.0, 10.0, 20.0, 30.0])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if not all_curves:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        M = np.vstack(all_curves)
        m = np.nanmean(M, axis=0)
        s = np.nanstd(M, axis=0)
        ax.plot(xs, m, color="C0", linewidth=2.0)
        ax.fill_between(xs, m - s, m + s, color="C0", alpha=0.25)
    ax.set_xlabel("Rejection rate (%)")
    ax.set_ylabel("Mean ranking error")
    ax.set_xticks(xs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_instability_distribution_ibin(sample_csv: str, out_path: str) -> None:
    df = pd.read_csv(sample_csv)
    v = pd.to_numeric(df["I_bin"], errors="coerce").dropna().to_numpy()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if len(v) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        hi = float(np.max(v))
        lo = float(np.min(v))
        pad = 0.05 * (hi - lo + 1e-9)
        ax.hist(v, bins=min(30, max(10, len(v) // 15)), range=(max(0, lo - pad), hi + pad), color="C0", edgecolor="0.4", linewidth=0.3)
    ax.set_xlabel(r"$I_{\mathrm{bin}}$")
    ax.set_ylabel("Number of samples")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_instability_vs_error(sample_csv: str, out_path: str) -> None:
    df = pd.read_csv(sample_csv)
    x = pd.to_numeric(df["I_bin"], errors="coerce")
    y = pd.to_numeric(df["error"], errors="coerce")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    m = x.notna() & y.notna()
    ax.scatter(x[m], y[m], s=8, alpha=0.35, color="C0", edgecolors="none", rasterized=True)
    ax.set_xlabel(r"$I_{\mathrm{bin}}$")
    ax.set_ylabel("Ranking error")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_rejection_curve(rej_csv: str, out_path: str) -> None:
    ys = rejection_series(rej_csv)
    xs = np.array([0.0, 10.0, 20.0, 30.0])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if ys is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(xs, ys, marker="o", color="C0", linewidth=1.5, markersize=5)
    ax.set_xlabel("Rejection rate (%)")
    ax.set_ylabel("Mean ranking error")
    ax.set_xticks(xs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=str, default="PromptAD/result_round1")
    parser.add_argument("--out_root", type=str, default="PromptAD/appendix_promptad_minimal")
    args = parser.parse_args()

    result_root = os.path.abspath(os.path.join(_repo_root(), args.result_root))
    out_root = os.path.abspath(os.path.join(_repo_root(), args.out_root))
    ec_dir = os.path.join(out_root, "extra_category")
    dk_dir = os.path.join(out_root, "different_k")
    for d in (out_root, ec_dir, dk_dir):
        os.makedirs(d, exist_ok=True)

    # --- A1 global: Spearman over all runs with exp5 ---
    spearmans: List[float] = []
    all_rej: List[np.ndarray] = []
    for info in iter_all_exp5_runs(result_root):
        df = pd.read_csv(info["path"])
        sp, _, _ = spearman_pearson(df)
        if sp == sp:
            spearmans.append(sp)
        rd = os.path.dirname(os.path.dirname(info["path"]))
        rj = os.path.join(rd, "experiments", "exp5_instability_rejection.csv")
        curve = rejection_series(rj)
        if curve is not None:
            all_rej.append(curve)

    plot_spearman_histogram(spearmans, os.path.join(out_root, "fig_s1_spearman_histogram.png"))
    plot_mean_rejection(all_rej, os.path.join(out_root, "fig_s2_mean_rejection.png"))

    # --- Extra category (capsules > pcb2 > candle) ---
    ds_e, cat_e, k_e, rn_e = pick_extra_category(result_root)
    rd_e = run_dir(result_root, ds_e, k_e, rn_e)
    samp_e = os.path.join(rd_e, "experiments", "exp5_sample_ranking_error.csv")
    rej_e = os.path.join(rd_e, "experiments", "exp5_instability_rejection.csv")
    plot_instability_distribution_ibin(samp_e, os.path.join(ec_dir, "instability_distribution.png"))
    plot_instability_vs_error(samp_e, os.path.join(ec_dir, "instability_vs_error.png"))
    plot_rejection_curve(rej_e, os.path.join(ec_dir, "rejection_curve.png"))

    # --- Different k: macaroni1 k4 or k8 ---
    ds_k, cat_k, k_k, rn_k = pick_diff_k(result_root)
    rd_k = run_dir(result_root, ds_k, k_k, rn_k)
    samp_k = os.path.join(rd_k, "experiments", "exp5_sample_ranking_error.csv")
    rej_k = os.path.join(rd_k, "experiments", "exp5_instability_rejection.csv")
    plot_instability_distribution_ibin(samp_k, os.path.join(dk_dir, "instability_distribution.png"))
    plot_instability_vs_error(samp_k, os.path.join(dk_dir, "instability_vs_error.png"))
    plot_rejection_curve(rej_k, os.path.join(dk_dir, "rejection_curve.png"))

    # --- table_summary.csv (two representative runs) ---
    def row_for_run(dataset: str, category: str, k: int, run_name: str) -> Dict[str, Any]:
        rd = run_dir(result_root, dataset, k, run_name)
        df = pd.read_csv(os.path.join(rd, "experiments", "exp5_sample_ranking_error.csv"))
        sp, _, _ = spearman_pearson(df)
        curve = rejection_series(os.path.join(rd, "experiments", "exp5_instability_rejection.csv"))
        if curve is None:
            b, e30 = float("nan"), float("nan")
        else:
            b, e30 = float(curve[0]), float(curve[3])
        rel = (b - e30) / b if b == b and abs(b) > 1e-12 else float("nan")
        return {
            "dataset": dataset,
            "category": category,
            "k": k,
            "spearman": sp,
            "baseline_error": b,
            "error_30": e30,
            "rel_drop_30": rel,
        }

    tbl = pd.DataFrame(
        [
            row_for_run(ds_e, cat_e, k_e, rn_e),
            row_for_run(ds_k, cat_k, k_k, rn_k),
        ]
    )
    tbl.to_csv(os.path.join(out_root, "table_summary.csv"), index=False)

    print("Wrote:", out_root)
    print("Extra category:", ds_e, cat_e, f"k={k_e}")
    print("Different k:", ds_k, cat_k, f"k={k_k}")
    print("Spearman runs:", len(spearmans), "Rejection curves:", len(all_rej))


if __name__ == "__main__":
    main()
