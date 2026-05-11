#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan")
    return float(x.mean())


def _median(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan")
    return float(x.median())


def _load_selection_csv(path: str, eps: float) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    need = [
        "dataset",
        "class",
        "k",
        "coverage",
        "auroc_auc_seed",
        "instability_auc_seed",
        "risk_auc_seed",
        "risk_stable_seed",
        "risk_oracle_seed",
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {path}: {miss}")

    for c in ["coverage", "auroc_auc_seed", "instability_auc_seed", "risk_auc_seed", "risk_stable_seed", "risk_oracle_seed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["epsilon"] = float(eps)
    return df


def _build_failure_analysis(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = pd.DataFrame()
    out["dataset"] = df["dataset"].astype(str)
    out["class"] = df["class"].astype(str)
    out["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")
    out["coverage"] = df["coverage"]
    out["epsilon"] = df["epsilon"]
    out["auroc_auc_seed"] = df["auroc_auc_seed"]
    out["instability_auc_seed"] = df["instability_auc_seed"]
    out["risk_auc_seed"] = df["risk_auc_seed"]
    out["risk_stable_seed"] = df["risk_stable_seed"]
    out["risk_oracle_seed"] = df["risk_oracle_seed"]
    out["delta_risk"] = out["risk_auc_seed"] - out["risk_stable_seed"]
    out["gap_auc"] = out["risk_auc_seed"] - out["risk_oracle_seed"]
    out["gap_stable"] = out["risk_stable_seed"] - out["risk_oracle_seed"]
    out["gap_reduction"] = out["gap_auc"] - out["gap_stable"]
    out["failure"] = out["gap_auc"] > float(threshold)
    out = out.dropna(
        subset=[
            "coverage",
            "epsilon",
            "auroc_auc_seed",
            "instability_auc_seed",
            "risk_auc_seed",
            "risk_stable_seed",
            "risk_oracle_seed",
            "gap_auc",
            "delta_risk",
            "gap_stable",
            "gap_reduction",
        ]
    ).reset_index(drop=True)
    return out


def _build_failure_summary(fa: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (eps, cov), g in fa.groupby(["epsilon", "coverage"], sort=True):
        g_f = g[g["failure"] == True]
        g_nf = g[g["failure"] == False]
        rows.append(
            {
                "epsilon": float(eps),
                "coverage": float(cov),
                "n_total": int(len(g)),
                "n_failure": int(len(g_f)),
                "failure_rate": float(np.mean(g["failure"])) if len(g) else float("nan"),
                "mean_instability_failure": _mean(g_f["instability_auc_seed"]),
                "median_instability_failure": _median(g_f["instability_auc_seed"]),
                "mean_instability_non_failure": _mean(g_nf["instability_auc_seed"]),
                "median_instability_non_failure": _median(g_nf["instability_auc_seed"]),
                "mean_delta_risk_all": _mean(g["delta_risk"]),
                "median_delta_risk_all": _median(g["delta_risk"]),
                "mean_delta_risk_failure": _mean(g_f["delta_risk"]),
                "median_delta_risk_failure": _median(g_f["delta_risk"]),
                "mean_gap_auc": _mean(g_f["gap_auc"]),
                "median_gap_auc": _median(g_f["gap_auc"]),
                "mean_gap_stable": _mean(g_f["gap_stable"]),
                "median_gap_stable": _median(g_f["gap_stable"]),
                "mean_gap_reduction": _mean(g_f["gap_reduction"]),
                "median_gap_reduction": _median(g_f["gap_reduction"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["epsilon", "coverage"]).reset_index(drop=True)


def _plot_scatter(fa: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    color = np.where(fa["failure"].to_numpy(dtype=bool), "tab:red", "tab:blue")
    ax.scatter(
        fa["instability_auc_seed"].to_numpy(dtype=np.float64),
        fa["gap_auc"].to_numpy(dtype=np.float64),
        c=color,
        alpha=0.7,
        s=28,
        edgecolors="none",
    )
    ax.set_xlabel("instability_auc_seed")
    ax.set_ylabel("gap_auc = risk_auc_seed - risk_oracle_seed")
    ax.set_title("Instability vs AUROC-selection Failure Gap")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_conditional_gain(fs: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for eps, g in fs.groupby("epsilon", sort=True):
        g = g.sort_values("coverage")
        x = g["coverage"].to_numpy(dtype=np.float64)
        y_all = g["mean_delta_risk_all"].to_numpy(dtype=np.float64)
        y_fail = g["mean_delta_risk_failure"].to_numpy(dtype=np.float64)
        ax.plot(x, y_all, marker="o", label=f"all, eps={eps:.3f}")
        ax.plot(x, y_fail, marker="s", linestyle="--", label=f"failure, eps={eps:.3f}")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("coverage")
    ax.set_ylabel("mean_delta_risk")
    ax.set_title("Conditional Gain: All vs Failure")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_oracle_gap_reduction(fs: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for eps, g in fs.groupby("epsilon", sort=True):
        g = g.sort_values("coverage")
        ax.plot(
            g["coverage"].to_numpy(dtype=np.float64),
            g["mean_gap_reduction"].to_numpy(dtype=np.float64),
            marker="o",
            label=f"eps={eps:.3f}",
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("coverage")
    ax.set_ylabel("mean_gap_reduction (failure only)")
    ax.set_title("Oracle Gap Reduction")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _print_key_conclusions(fs: pd.DataFrame, min_failure: int) -> None:
    print("=== Failure-driven analysis conclusions ===")
    for (eps, cov), r in fs.groupby(["epsilon", "coverage"], sort=True):
        row = r.iloc[0]
        n_f = int(row["n_failure"])
        n_t = int(row["n_total"])
        fr = float(row["failure_rate"])
        mi_f = float(row["mean_instability_failure"])
        mi_nf = float(row["mean_instability_non_failure"])
        mdi_f = float(row["median_instability_failure"])
        mdi_nf = float(row["median_instability_non_failure"])
        md_all = float(row["mean_delta_risk_all"])
        md_fail = float(row["mean_delta_risk_failure"])
        med_all = float(row["median_delta_risk_all"])
        med_fail = float(row["median_delta_risk_failure"])
        mgr = float(row["mean_gap_reduction"])
        mdgr = float(row["median_gap_reduction"])

        print(f"\n[epsilon={eps:.3f}, coverage={cov:.1f}]")
        print(f"Failure rate: {fr*100:.2f}% ({n_f}/{n_t})")
        print(
            "Instability (failure vs non-failure): "
            f"mean {mi_f:.6f} vs {mi_nf:.6f}; median {mdi_f:.6f} vs {mdi_nf:.6f}"
        )
        print(
            "Mean delta risk (all vs failure): "
            f"mean {md_all:.6f} vs {md_fail:.6f}; median {med_all:.6f} vs {med_fail:.6f}"
        )
        print(f"Mean oracle gap reduction: {mgr:.6f} (median {mdgr:.6f})")

        instab_higher = np.isfinite(mi_f) and np.isfinite(mi_nf) and (mi_f > mi_nf)
        gain_better = np.isfinite(md_fail) and np.isfinite(md_all) and (md_fail > md_all)
        oracle_closer = np.isfinite(mgr) and (mgr > 0)
        print(
            "Judgement: "
            f"instability_higher_in_failure={instab_higher}, "
            f"stable_more_effective_in_failure={gain_better}, "
            f"closer_to_oracle={oracle_closer}"
        )
        if n_f < int(min_failure):
            print(f"WARNING: failure sample too small (n={n_f} < {int(min_failure)})")


def run(args: argparse.Namespace) -> None:
    input_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dfs = []
    for eps in args.epsilons:
        p = os.path.join(input_dir, f"selection_summary_epsilon_{eps:.3f}.csv")
        dfs.append(_load_selection_csv(p, eps=float(eps)))
    merged = pd.concat(dfs, axis=0, ignore_index=True)

    # Main threshold.
    fa = _build_failure_analysis(merged, threshold=float(args.threshold))
    fa.to_csv(os.path.join(out_dir, "failure_analysis.csv"), index=False)
    fs = _build_failure_summary(fa)
    fs.to_csv(os.path.join(out_dir, "failure_summary.csv"), index=False)

    _plot_scatter(fa, os.path.join(out_dir, "failure_instability_scatter.png"))
    _plot_conditional_gain(fs, os.path.join(out_dir, "conditional_gain.png"))
    _plot_oracle_gap_reduction(fs, os.path.join(out_dir, "oracle_gap_reduction.png"))

    # Robustness threshold output.
    if args.robust_threshold is not None:
        fa_r = _build_failure_analysis(merged, threshold=float(args.robust_threshold))
        fs_r = _build_failure_summary(fa_r)
        fa_r.to_csv(os.path.join(out_dir, f"failure_analysis_thr_{args.robust_threshold:.3f}.csv"), index=False)
        fs_r.to_csv(os.path.join(out_dir, f"failure_summary_thr_{args.robust_threshold:.3f}.csv"), index=False)

    _print_key_conclusions(fs, min_failure=int(args.min_failure))
    print(f"\nWrote outputs to: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Failure-driven analysis for AUROC selection failures.")
    p.add_argument(
        "--input-dir",
        type=str,
        default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/pilot_instability_selection",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/pilot_instability_selection/failure_driven",
    )
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.002, 0.005])
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--robust-threshold", type=float, default=0.02)
    p.add_argument("--min-failure", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
