"""
Appendix A.1 (PromptAD empirical observation): aggregate all runs under result_round1,
read-only. Writes PromptAD/appendix_promptad/ (figures + tables).

Does not modify any experiment outputs.
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

# Unified figure style (no decorative colormaps)
FIG_W, FIG_H = 7.0, 5.5
DPI = 300
SCATTER_COLOR = "0.35"
LINE_COLOR = "0.25"
HIST_COLOR = "0.45"


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
        sample_csv = os.path.join(exp_dir, "exp5_sample_ranking_error.csv")
        rej_csv = os.path.join(exp_dir, "exp5_instability_rejection.csv")
        runs.append(
            {
                "dataset": dataset,
                "category": category,
                "k": k,
                "run_name": run_name,
                "run_dir": run_dir,
                "summary_path": summary_path,
                "sample_csv": sample_csv,
                "rejection_csv": rej_csv,
            }
        )
    return runs


def enrich_run_meta(r: Dict[str, Any]) -> Dict[str, Any]:
    d = load_summary(r["summary_path"])
    r = dict(r)
    r["auroc"] = auroc_from_summary(d)
    r["instability"] = instability_from_summary(d)
    return r


def save_scatter_auroc_instability(
    df: pd.DataFrame,
    out_path: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sub = df.dropna(subset=["auroc", "instability"])
    ax.scatter(
        sub["auroc"],
        sub["instability"],
        s=22,
        alpha=0.55,
        c=SCATTER_COLOR,
        edgecolors="none",
    )
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Instability (mean flip rate / I summary)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_histogram(values: np.ndarray, out_path: str, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        vmax = float(np.max(vals))
        upper = min(1.0, vmax + 0.05 * (vmax + 1e-6))
        ax.hist(vals, bins=min(30, max(10, len(vals) // 3)), range=(0.0, upper), color=HIST_COLOR, edgecolor="0.2", linewidth=0.4)
        ax.set_xlim(0, upper)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count (runs)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def corr_spearman_pearson(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    m = x.notna() & y.notna()
    xx = x[m].astype(float)
    yy = y[m].astype(float)
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    if float(np.nanstd(xx.to_numpy())) < 1e-12 or float(np.nanstd(yy.to_numpy())) < 1e-12:
        return float("nan"), float("nan"), n
    sp = float(xx.corr(yy, method="spearman"))
    pe = float(xx.corr(yy, method="pearson"))
    return sp, pe, n


def save_scatter_ibin_error(x: np.ndarray, y: np.ndarray, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.scatter(x, y, s=6, alpha=0.25, c=SCATTER_COLOR, edgecolors="none", rasterized=True)
    ax.set_xlabel(r"$I_{\mathrm{bin}}$")
    ax.set_ylabel("Ranking error")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def rejection_rows_from_csv(path: str) -> Dict[str, float]:
    df = pd.read_csv(path, comment="#")
    out: Dict[str, float] = {}
    for key, col in [
        ("baseline", "baseline"),
        ("reject_10", "reject_10"),
        ("reject_20", "reject_20"),
        ("reject_30", "reject_30"),
    ]:
        row = df.loc[df["setting"] == col, "mean_error_all"]
        if not row.empty:
            out[key] = float(row.iloc[0])
    return out


def monotonic_nonincreasing(vals: List[float]) -> bool:
    for a, b in zip(vals, vals[1:]):
        if b > a + 1e-9:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_root",
        type=str,
        default="PromptAD/result_round1",
        help="Root containing dataset/k_*/pairwise_instability/...",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="PromptAD/appendix_promptad",
    )
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(_repo_root(), args.result_root))
    out_root = os.path.abspath(os.path.join(_repo_root(), args.out_root))

    a1 = os.path.join(out_root, "A1_auroc_vs_instability")
    a2 = os.path.join(out_root, "A2_instability_distribution")
    a3 = os.path.join(out_root, "A3_instability_vs_error")
    a4 = os.path.join(out_root, "A4_rejection")
    tables = os.path.join(out_root, "tables")
    for d in (a1, a2, a3, a4, tables):
        os.makedirs(d, exist_ok=True)

    runs = discover_runs(root)
    if not runs:
        raise SystemExit(f"No runs found under {root}")

    rows_meta: List[Dict[str, Any]] = []
    for r in runs:
        rows_meta.append(enrich_run_meta(r))
    meta = pd.DataFrame(rows_meta)

    # ---------------- A1 ----------------
    save_scatter_auroc_instability(meta, os.path.join(a1, "fig_s1_all.png"), "AUROC vs instability (all runs)")
    save_scatter_auroc_instability(
        meta[meta["dataset"] == "mvtec"],
        os.path.join(a1, "fig_s2_mvtec.png"),
        "AUROC vs instability (MVTec AD)",
    )
    save_scatter_auroc_instability(
        meta[meta["dataset"] == "visa"],
        os.path.join(a1, "fig_s3_visa.png"),
        "AUROC vs instability (VisA)",
    )
    ks = sorted(meta["k"].unique())
    s_idx = 4
    for k in ks:
        save_scatter_auroc_instability(
            meta[meta["k"] == k],
            os.path.join(a1, f"fig_s{s_idx}_k{k}.png"),
            f"AUROC vs instability (k={k})",
        )
        s_idx += 1

    # ---------------- A2 ----------------
    inst = meta["instability"].to_numpy(dtype=float)
    save_histogram(inst, os.path.join(a2, "fig_s6_all.png"), "Run-level instability (all)", "Mean instability")
    save_histogram(
        meta.loc[meta["dataset"] == "mvtec", "instability"].to_numpy(dtype=float),
        os.path.join(a2, "fig_s7_mvtec.png"),
        "Run-level instability (MVTec AD)",
        "Mean instability",
    )
    save_histogram(
        meta.loc[meta["dataset"] == "visa", "instability"].to_numpy(dtype=float),
        os.path.join(a2, "fig_s8_visa.png"),
        "Run-level instability (VisA)",
        "Mean instability",
    )

    dist_out = meta[
        ["dataset", "category", "k", "run_name", "auroc", "instability"]
    ].rename(columns={"instability": "mean_instability"})
    dist_out.to_csv(os.path.join(tables, "instability_distribution.csv"), index=False)

    # Per-category histograms (appendix: full coverage)
    for ds in sorted(meta["dataset"].unique()):
        sub_ds = meta[meta["dataset"] == ds]
        cats = sorted(sub_ds["category"].unique())
        n = len(cats)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_W * 0.9 * ncols / 2, FIG_H * 0.85 * nrows / 2))
        axes_flat = np.atleast_1d(axes).ravel()
        for i, cat in enumerate(cats):
            ax = axes_flat[i]
            v = sub_ds.loc[sub_ds["category"] == cat, "instability"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                ax.set_axis_off()
                continue
            vmax = float(np.max(v))
            upper = min(1.0, vmax + 0.05 * (vmax + 1e-6))
            ax.hist(v, bins=min(15, max(5, len(v))), range=(0.0, upper), color=HIST_COLOR, edgecolor="0.2", linewidth=0.3)
            ax.set_title(cat, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for j in range(len(cats), len(axes_flat)):
            axes_flat[j].set_axis_off()
        fig.suptitle(f"Instability by category ({ds})", fontsize=11)
        fig.tight_layout()
        fig.savefig(
            os.path.join(a2, f"fig_hist_by_category_{ds}.png"),
            dpi=DPI,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ---------------- A3 ----------------
    corr_rows: List[Dict[str, Any]] = []
    all_ix: List[np.ndarray] = []
    all_iy: List[np.ndarray] = []

    for _, r in meta.iterrows():
        p = r["sample_csv"]
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p)
        if "I_bin" not in df.columns or "error" not in df.columns:
            continue
        sp, pe, n = corr_spearman_pearson(df["I_bin"], df["error"])
        corr_rows.append(
            {
                "dataset": r["dataset"],
                "category": r["category"],
                "k": int(r["k"]),
                "spearman": sp,
                "pearson": pe,
                "num_samples": n,
            }
        )
        all_ix.append(df["I_bin"].to_numpy(dtype=float))
        all_iy.append(df["error"].to_numpy(dtype=float))

    corr_df = pd.DataFrame(corr_rows)
    corr_df = corr_df[["dataset", "category", "k", "spearman", "pearson", "num_samples"]]
    corr_df.to_csv(os.path.join(tables, "instability_error_correlation.csv"), index=False)

    if all_ix:
        x_all = np.concatenate(all_ix)
        y_all = np.concatenate(all_iy)
        save_scatter_ibin_error(x_all, y_all, os.path.join(a3, "fig_s9_all.png"), "Instability vs error (all samples, all runs)")

    for ds in sorted(meta["dataset"].unique()):
        ix_list: List[np.ndarray] = []
        iy_list: List[np.ndarray] = []
        for _, r in meta[meta["dataset"] == ds].iterrows():
            p = r["sample_csv"]
            if not os.path.isfile(p):
                continue
            df = pd.read_csv(p)
            if "I_bin" not in df.columns or "error" not in df.columns:
                continue
            ix_list.append(df["I_bin"].to_numpy(dtype=float))
            iy_list.append(df["error"].to_numpy(dtype=float))
        if not ix_list:
            continue
        tag = "s10_mvtec" if ds == "mvtec" else "s11_visa"
        save_scatter_ibin_error(
            np.concatenate(ix_list),
            np.concatenate(iy_list),
            os.path.join(a3, f"fig_{tag}.png"),
            f"Instability vs error ({ds}, all categories)",
        )

    fig_s = 12
    for ds in sorted(meta["dataset"].unique()):
        for cat in sorted(meta.loc[meta["dataset"] == ds, "category"].unique()):
            ix_list = []
            iy_list = []
            sel = meta[(meta["dataset"] == ds) & (meta["category"] == cat)]
            for _, r in sel.iterrows():
                p = r["sample_csv"]
                if not os.path.isfile(p):
                    continue
                df = pd.read_csv(p)
                if "I_bin" not in df.columns or "error" not in df.columns:
                    continue
                ix_list.append(df["I_bin"].to_numpy(dtype=float))
                iy_list.append(df["error"].to_numpy(dtype=float))
            if not ix_list:
                continue
            safe_cat = cat.replace("/", "_").replace(" ", "_")
            save_scatter_ibin_error(
                np.concatenate(ix_list),
                np.concatenate(iy_list),
                os.path.join(a3, f"fig_s{fig_s}_{ds}_{safe_cat}.png"),
                f"Instability vs error ({ds} / {cat})",
            )
            fig_s += 1

    # ---------------- A4 ----------------
    rej_rows: List[Dict[str, Any]] = []

    for _, r in meta.iterrows():
        p = r["rejection_csv"]
        if not os.path.isfile(p):
            continue
        try:
            er = rejection_rows_from_csv(p)
        except Exception:
            continue
        if len(er) < 4:
            continue
        b = er["baseline"]
        e10 = er["reject_10"]
        e20 = er["reject_20"]
        e30 = er["reject_30"]
        rel_drop = (b - e30) / b if abs(b) > 1e-12 else float("nan")
        mono = monotonic_nonincreasing([b, e10, e20, e30])
        rej_rows.append(
            {
                "dataset": r["dataset"],
                "category": r["category"],
                "k": int(r["k"]),
                "baseline_error": b,
                "error_10": e10,
                "error_20": e20,
                "error_30": e30,
                "rel_drop_30": rel_drop,
                "is_monotonic": mono,
            }
        )
    rej_df = pd.DataFrame(rej_rows)
    rej_df = rej_df[
        [
            "dataset",
            "category",
            "k",
            "baseline_error",
            "error_10",
            "error_20",
            "error_30",
            "rel_drop_30",
            "is_monotonic",
        ]
    ]
    rej_df.to_csv(os.path.join(tables, "rejection_summary.csv"), index=False)

    xs = np.array([0.0, 10.0, 20.0, 30.0])

    def plot_overlay(group_filter: Optional[str], out_path: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        mat: List[np.ndarray] = []
        for _, r in meta.iterrows():
            if group_filter and r["dataset"] != group_filter:
                continue
            p = r["rejection_csv"]
            if not os.path.isfile(p):
                continue
            try:
                er = rejection_rows_from_csv(p)
            except Exception:
                continue
            if len(er) < 4:
                continue
            ys = np.array([er["baseline"], er["reject_10"], er["reject_20"], er["reject_30"]], dtype=float)
            mat.append(ys)
            ax.plot(xs, ys, color=LINE_COLOR, alpha=0.18, linewidth=0.9)
        if mat:
            M = np.vstack(mat)
            m = np.nanmean(M, axis=0)
            ax.plot(xs, m, color="0.05", linewidth=2.0, label="Mean over runs")
            ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.set_xlabel("Rejection rate (%)")
        ax.set_ylabel("Mean ranking error")
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    plot_overlay(None, os.path.join(a4, "fig_s20_all.png"), "Rejection curves (all runs)")
    plot_overlay("mvtec", os.path.join(a4, "fig_s21_mvtec.png"), "Rejection curves (MVTec AD)")
    plot_overlay("visa", os.path.join(a4, "fig_s22_visa.png"), "Rejection curves (VisA)")

    # Mean-only curve (no per-run spaghetti)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    mat: List[np.ndarray] = []
    for _, r in meta.iterrows():
        p = r["rejection_csv"]
        if not os.path.isfile(p):
            continue
        try:
            er = rejection_rows_from_csv(p)
        except Exception:
            continue
        if len(er) < 4:
            continue
        mat.append(
            np.array([er["baseline"], er["reject_10"], er["reject_20"], er["reject_30"]], dtype=float)
        )
    if mat:
        M = np.vstack(mat)
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        ax.plot(xs, mean, color="0.1", linewidth=2.0, label="Mean")
        ax.fill_between(xs, mean - std, mean + std, color="0.5", alpha=0.25, label="±1 std")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlabel("Rejection rate (%)")
    ax.set_ylabel("Mean ranking error")
    ax.set_title("Average rejection curve (all runs)")
    ax.set_xticks(xs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a4, "fig_s23_mean.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("Wrote appendix under:", out_root)
    print("Runs (summary) discovered:", len(meta))
    print("Correlation rows (with exp5):", len(corr_df))
    print("Rejection rows (complete exp5):", len(rej_df))


if __name__ == "__main__":
    main()
