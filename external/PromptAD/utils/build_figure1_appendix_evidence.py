"""
Evidence tables and figures supporting main-text Figure 1 (PromptAD), read-only.

- (a)(b): per-run AUROC + flip-rate (instability) tables + distribution summary for (b).
- (c): Spearman(I_bin, error) per run → histogram across all category–k settings with exp5.
- (d): per-run rejection effect + aggregate counts + histogram of relative drop at 30% rejection.

Writes under PromptAD/appendix_promptad_minimal/ (tables/ + figure1_evidence/).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGSIZE = (6.5, 4.5)
DPI = 300


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse_k(folder_name: str) -> Optional[int]:
    if folder_name.startswith("k_"):
        try:
            return int(folder_name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def parse_category_from_run(run_name: str, dataset: str, k: int) -> str:
    prefix = f"CLS-{dataset}-"
    suffix = f"-k{k}-"
    if run_name.startswith(prefix) and suffix in run_name:
        return run_name[len(prefix) : run_name.index(suffix)]
    return run_name


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def auroc_from_summary(d: Dict[str, Any]) -> float:
    for key in ("sklearn_auroc_final", "pairwise_auroc_final", "auroc", "AUROC"):
        if key in d and d[key] is not None:
            return float(d[key])
    return float("nan")


def instability_from_summary(d: Dict[str, Any]) -> float:
    for key in ("mean_I_bin", "flip_rate_mean", "anomaly_I_bin_mean"):
        if key in d and d[key] is not None:
            return float(d[key])
    return float("nan")


def discover_runs(result_root: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(result_root, "*", "k_*", "pairwise_instability", "*", "summary.json")
    runs: List[Dict[str, Any]] = []
    for summary_path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(summary_path)
        rel = os.path.relpath(summary_path, result_root)
        parts = rel.split(os.sep)
        if len(parts) < 4:
            continue
        dataset = parts[0]
        k = _parse_k(parts[1])
        if k is None:
            continue
        run_name = parts[3]
        category = parse_category_from_run(run_name, dataset, k)
        exp_dir = os.path.join(run_dir, "experiments")
        runs.append(
            {
                "dataset": dataset,
                "category": category,
                "k": k,
                "run_name": run_name,
                "run_dir": run_dir,
                "summary_path": summary_path,
                "sample_csv": os.path.join(exp_dir, "exp5_sample_ranking_error.csv"),
                "rejection_csv": os.path.join(exp_dir, "exp5_instability_rejection.csv"),
            }
        )
    return runs


def spearman_ibin_error(df: pd.DataFrame) -> float:
    if "I_bin" not in df.columns or "error" not in df.columns:
        return float("nan")
    x = df["I_bin"].astype(float)
    y = df["error"].astype(float)
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 3:
        return float("nan")
    xx = x[m]
    yy = y[m]
    if float(np.nanstd(xx.to_numpy())) < 1e-12 or float(np.nanstd(yy.to_numpy())) < 1e-12:
        return float("nan")
    return float(xx.corr(yy, method="spearman"))


def rejection_metrics(rej_csv: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "baseline_error": float("nan"),
        "error_30": float("nan"),
        "rel_drop_30": float("nan"),
        "is_monotonic": False,
    }
    if not os.path.isfile(rej_csv):
        return out
    df = pd.read_csv(rej_csv, comment="#")
    order = ["baseline", "reject_10", "reject_20", "reject_30"]
    ys = []
    for s in order:
        row = df.loc[df["setting"] == s, "mean_error_all"]
        if row.empty:
            return out
        ys.append(float(row.iloc[0]))
    b, e10, e20, e30 = ys
    out["baseline_error"] = b
    out["error_30"] = e30
    if abs(b) > 1e-15:
        out["rel_drop_30"] = (b - e30) / b
    out["is_monotonic"] = e10 <= b + 1e-12 and e20 <= e10 + 1e-12 and e30 <= e20 + 1e-12
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=str, default="PromptAD/result_round1")
    parser.add_argument(
        "--out_minimal",
        type=str,
        default="PromptAD/appendix_promptad_minimal",
    )
    args = parser.parse_args()

    result_root = os.path.abspath(os.path.join(_repo_root(), args.result_root))
    minimal = os.path.abspath(os.path.join(_repo_root(), args.out_minimal))
    tables = os.path.join(minimal, "tables")
    evdir = os.path.join(minimal, "figure1_evidence")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(evdir, exist_ok=True)

    runs = discover_runs(result_root)
    rows_ab: List[Dict[str, Any]] = []
    rows_c: List[Dict[str, Any]] = []
    rows_d: List[Dict[str, Any]] = []

    for r in runs:
        d = load_summary(r["summary_path"])
        auroc = auroc_from_summary(d)
        inst = instability_from_summary(d)
        rows_ab.append(
            {
                "dataset": r["dataset"],
                "category": r["category"],
                "k": r["k"],
                "run_name": r["run_name"],
                "auroc": auroc,
                "flip_rate_mean": inst,
            }
        )

        sp = float("nan")
        n_samples = 0
        if os.path.isfile(r["sample_csv"]):
            df = pd.read_csv(r["sample_csv"])
            n_samples = int(len(df))
            sp = spearman_ibin_error(df)
        rows_c.append(
            {
                "dataset": r["dataset"],
                "category": r["category"],
                "k": r["k"],
                "run_name": r["run_name"],
                "num_samples": n_samples,
                "spearman_I_bin_error": sp,
            }
        )

        rm = rejection_metrics(r["rejection_csv"])
        rel = rm["rel_drop_30"]
        improved = rel == rel and rel > 1e-12
        rows_d.append(
            {
                "dataset": r["dataset"],
                "category": r["category"],
                "k": r["k"],
                "run_name": r["run_name"],
                "baseline_error": rm["baseline_error"],
                "error_30": rm["error_30"],
                "rel_drop_30": rel,
                "is_monotonic": rm["is_monotonic"],
                "improved_at_30pct": improved,
            }
        )

    df_ab = pd.DataFrame(rows_ab)
    df_c = pd.DataFrame(rows_c)
    df_d = pd.DataFrame(rows_d)

    # --- (a) detailed table: every setting for scatter in Fig.1(a)
    df_ab.sort_values(["dataset", "category", "k"]).to_csv(
        os.path.join(tables, "table_figure1_a_auroc_instability_all_runs.csv"),
        index=False,
    )

    # --- (b) same per-run flip rates + distribution summary (histogram in Fig.1(b))
    df_ab.rename(columns={"flip_rate_mean": "pairwise_flip_rate_mean"}).to_csv(
        os.path.join(tables, "table_figure1_b_flip_rate_per_run.csv"),
        index=False,
    )
    vals = df_ab["flip_rate_mean"].dropna().astype(float)
    summary_b = pd.DataFrame(
        [
            {
                "n_runs": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()) if len(vals) else float("nan"),
                "q25": float(vals.quantile(0.25)) if len(vals) else float("nan"),
                "median": float(vals.median()) if len(vals) else float("nan"),
                "q75": float(vals.quantile(0.75)) if len(vals) else float("nan"),
                "max": float(vals.max()) if len(vals) else float("nan"),
            }
        ]
    )
    summary_b.to_csv(os.path.join(tables, "table_figure1_b_distribution_summary.csv"), index=False)

    # --- (c) per-run Spearman + histogram
    df_c.sort_values(["dataset", "category", "k"]).to_csv(
        os.path.join(tables, "table_figure1_c_spearman_per_run.csv"),
        index=False,
    )
    sp_finite = df_c["spearman_I_bin_error"].dropna().astype(float)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if len(sp_finite):
        ax.hist(sp_finite, bins=min(22, max(8, len(sp_finite) // 3)), color="C0", edgecolor="0.35", linewidth=0.4)
        sp_mean = float(sp_finite.mean())
        ax.axvline(sp_mean, color="0.1", linestyle="-", linewidth=1.6, zorder=5)
        tx = ax.text(
            0.97,
            1.06,
            f"mean = {sp_mean:.3f}\n(n = {len(sp_finite)})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="0.1",
            zorder=6,
        )
        tx.set_clip_on(False)
    ax.set_xlabel(r"Spearman $\rho$ ($I_{\mathrm{bin}}$, ranking error)")
    ax.set_ylabel("Number of runs (dataset–category–k)")
    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=0.8, zorder=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(evdir, "fig_figure1_c_spearman_distribution.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # --- (d) per-run rejection + aggregate + figures
    df_d.sort_values(["dataset", "category", "k"]).to_csv(
        os.path.join(tables, "table_figure1_d_rejection_per_run.csv"),
        index=False,
    )

    valid_rel = df_d["rel_drop_30"].notna()
    n_valid = int(valid_rel.sum())
    n_improved = int((df_d.loc[valid_rel, "rel_drop_30"] > 1e-12).sum())
    n_strict_non_worse = int((df_d.loc[valid_rel, "rel_drop_30"] >= -1e-12).sum())
    n_worse = n_valid - n_strict_non_worse
    n_mono_defined = int(df_d.loc[valid_rel, "is_monotonic"].sum())
    n_total_runs = int(len(df_d))

    agg = pd.DataFrame(
        [
            {
                "n_runs_summary_level": n_total_runs,
                "n_runs_with_defined_rel_drop_30": n_valid,
                "n_improved_rel_drop_30_strictly_positive": n_improved,
                "fraction_improved_of_defined": (n_improved / n_valid) if n_valid else float("nan"),
                "n_non_worse_rel_drop_30_ge_0": n_strict_non_worse,
                "n_worse_rel_drop_30_lt_0": n_worse,
                "fraction_non_worse_of_defined": (n_strict_non_worse / n_valid) if n_valid else float("nan"),
                "n_monotonic_among_defined_rel_drop": n_mono_defined,
                "fraction_monotonic_of_defined": (n_mono_defined / n_valid) if n_valid else float("nan"),
                "median_rel_drop_30_over_defined": float(df_d.loc[valid_rel, "rel_drop_30"].median()) if n_valid else float("nan"),
            }
        ]
    )
    agg.to_csv(os.path.join(tables, "table_figure1_d_aggregate_summary.csv"), index=False)

    rel_vals = df_d.loc[valid_rel, "rel_drop_30"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if len(rel_vals):
        ax.hist(rel_vals, bins=min(24, max(8, len(rel_vals) // 3)), color="C0", edgecolor="0.35", linewidth=0.4)
    ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Relative error drop at 30% rejection: $(e_0 - e_{30}) / e_0$")
    ax.set_ylabel("Number of runs")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(evdir, "fig_figure1_d_rel_drop_histogram.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Bar: worse vs non-worse (among defined rel_drop)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    labels = ["Rel. drop < 0\n(error increases)", "Rel. drop >= 0\n(same or lower error)"]
    counts = [n_worse, n_strict_non_worse]
    ax.bar(labels, counts, color=["C1", "C0"], edgecolor="0.2", linewidth=0.5)
    ax.set_ylabel("Number of runs (defined rel. drop)")
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.02, str(c), ha="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(evdir, "fig_figure1_d_improved_counts.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("Tables:", tables)
    print("Figures:", evdir)
    print("Runs (summary):", len(runs), "| Spearman finite:", int(sp_finite.notna().sum()), "| rel_drop defined:", n_valid)


if __name__ == "__main__":
    main()
