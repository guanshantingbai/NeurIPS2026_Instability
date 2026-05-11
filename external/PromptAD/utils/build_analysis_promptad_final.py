"""
NeurIPS Figure 1 (final): single 2x2 matplotlib figure + standalone panel (a).

Panel (d) is drawn in matplotlib like (a–c): risk–coverage curves from the same
per-sample CSVs as seed_killer_evidence_pipeline (killer_pair.json + proxy u6).

Outputs:
  figures/Analysis_PromptAD_final.{png,pdf}
  figures/figure_a_counterexample.png
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROMPTAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROMPTAD_DIR not in sys.path:
    sys.path.insert(0, _PROMPTAD_DIR)

from utils.rejection_instability_analysis import build_analysis_frame, load_and_validate_csv
from utils.seed_killer_evidence_pipeline import (
    PROXY_COL,
    default_coverage_grid,
    find_per_sample_csv,
    risk_coverage,
)

_LOG_PANEL_D = logging.getLogger("build_analysis_promptad_final.panel_d")
_LOG_PANEL_D.handlers.clear()
_LOG_PANEL_D.addHandler(logging.NullHandler())
_LOG_PANEL_D.propagate = False

TITLE_FS = 10
AXIS_FS = 10
TICK_FS = 9
ANNOT_FS = 9
ANNOT_FS_PANEL_A = 8
PANEL_LETTER_FS = 11
TREND_COLOR = "#D98880"
PNG_DPI = 450


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_auroc_for_row(result_root: str, row: pd.Series) -> float:
    p = os.path.join(
        result_root,
        str(row["dataset"]),
        f"k_{int(row['k'])}",
        "pairwise_instability",
        str(row["run_name"]),
        "summary.json",
    )
    if not os.path.isfile(p):
        return float("nan")
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        for key in ("sklearn_auroc_final", "pairwise_auroc_final", "auroc", "AUROC"):
            if key in d and d[key] is not None:
                return float(d[key])
    except Exception:
        pass
    return float("nan")


def _instability_column(df: pd.DataFrame) -> str:
    for c in ("mean_I_bin", "mean_ibin", "I_bin_mean", "flip_rate_mean"):
        if c in df.columns:
            return c
    raise ValueError("No instability column found (expected mean_I_bin or flip_rate_mean)")


def load_enriched_table(exp_summary_csv: str, result_root: str) -> pd.DataFrame:
    df = pd.read_csv(exp_summary_csv).copy()
    df["AUROC"] = df.apply(lambda r: _load_auroc_for_row(result_root, r), axis=1)
    icol = _instability_column(df)
    df["instability"] = pd.to_numeric(df[icol], errors="coerce")
    df = df.dropna(subset=["AUROC", "instability"])
    df["AUROC"] = df["AUROC"].astype(float)
    df["instability"] = df["instability"].astype(float)
    return df


def _category_only_label(row: pd.Series, max_len: int = 14) -> str:
    """Short in-panel label (category name only) to reduce clutter and edge bleed."""
    cat = str(row.get("category", "")).strip().replace("_", " ")
    if len(cat) > max_len:
        return cat[: max_len - 1] + "\u2026"
    return cat or "—"


def draw_panel_a_counterexample(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Panel (a): comparable AUROC with a large instability gap (single scatter layer, no band shading).
    """
    x_all = df["AUROC"].to_numpy()
    y_all = df["instability"].to_numpy()

    ax.scatter(
        x_all,
        y_all,
        alpha=0.45,
        s=28,
        edgecolors="none",
        color="0.35",
        zorder=1,
    )

    band = df[(df["AUROC"] >= 0.9) & (df["AUROC"] <= 0.95)].copy()
    if len(band) < 2:
        band = df[(df["AUROC"] >= 0.88) & (df["AUROC"] <= 0.97)].copy()
    if len(band) < 2:
        band = df[(df["AUROC"] >= 0.85) & (df["AUROC"] <= 0.99)].copy()

    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
    ypad = max(1e-6, (ymax - ymin) * 0.08)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    if len(band) >= 2:
        band2 = band.sort_values("instability")
        low = band2.iloc[0]
        high = band2.iloc[-1]

        xl, yl = float(low["AUROC"]), float(low["instability"])
        xh, yh = float(high["AUROC"]), float(high["instability"])
        delta = yh - yl

        ax.scatter(
            [xl],
            [yl],
            s=135,
            c="blue",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        ax.scatter(
            [xh],
            [yh],
            s=135,
            c="red",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )

        low_lab = _category_only_label(low) if "category" in df.columns else "low"
        high_lab = _category_only_label(high) if "category" in df.columns else "high"
        afs = ANNOT_FS_PANEL_A
        ax.annotate(
            low_lab,
            (xl, yl),
            textcoords="offset points",
            xytext=(-32, -14),
            fontsize=afs,
            color="0.15",
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.5),
            zorder=6,
            clip_on=True,
        )
        # High-instability point: offset left/down to stay inside (a) and clear subplot gap to (b)
        ax.annotate(
            high_lab,
            (xh, yh),
            textcoords="offset points",
            xytext=(-26, -22),
            fontsize=afs,
            color="0.15",
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.5),
            zorder=6,
            clip_on=True,
        )

        ax.annotate(
            "",
            xy=(xh, yh),
            xytext=(xl, yl),
            arrowprops=dict(
                arrowstyle="->",
                color="0.25",
                lw=1.5,
                shrinkA=14,
                shrinkB=14,
            ),
            zorder=4,
            clip_on=True,
        )

        ax.text(
            0.97,
            0.97,
            f"Δ ≈ {delta:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=ANNOT_FS_PANEL_A,
            color="0.2",
            zorder=7,
        )

    ax.set_xlabel("AUROC", fontsize=AXIS_FS)
    ax.set_ylabel("Instability score", fontsize=AXIS_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_b_flip_hist(ax: plt.Axes, df: pd.DataFrame) -> None:
    vals = pd.to_numeric(df["flip_rate_mean"], errors="coerce").dropna().to_numpy()
    mean_v = float(np.mean(vals))
    max_v = float(np.max(vals))
    xmax = min(1.0, max_v + 0.06)
    ax.hist(vals, bins=15, range=(0.0, xmax), color="0.45", edgecolor="0.15", linewidth=0.5)
    ax.axvline(
        mean_v,
        color="red",
        linestyle="-",
        linewidth=1.0,
        label=f"mean flip rate = {mean_v:.2f}",
    )
    ax.set_xlim(0, xmax)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.06, 1.0),
        borderaxespad=0,
        frameon=True,
        fancybox=False,
        edgecolor="0.5",
        fontsize=TICK_FS,
    )
    ax.set_xlabel("Pairwise flip rate", fontsize=AXIS_FS)
    ax.set_ylabel("Number of settings", fontsize=AXIS_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_c_instability_error(ax: plt.Axes, sample_csv: str) -> None:
    d = pd.read_csv(sample_csv)
    if "I_bin" not in d.columns or "error" not in d.columns:
        raise ValueError(f"{sample_csv} needs I_bin and error columns")
    x = d["I_bin"].astype(float)
    y = d["error"].astype(float)
    sp = float(x.corr(y, method="spearman"))

    ax.scatter(x, y, alpha=0.28, s=8, edgecolors="none", rasterized=True, color="0.35", zorder=2)

    mask = x.notna() & y.notna()
    xx = x[mask].to_numpy(dtype=float)
    yy = y[mask].to_numpy(dtype=float)
    if len(xx) >= 2 and float(np.std(xx)) > 1e-12:
        coef = np.polyfit(xx, yy, 1)
        xs = np.linspace(float(np.min(xx)), float(np.max(xx)), 80)
        ax.plot(
            xs,
            np.polyval(coef, xs),
            color=TREND_COLOR,
            linewidth=1.35,
            alpha=0.62,
            zorder=1,
        )

    if sp == sp:
        ax.text(
            0.97,
            0.97,
            rf"Spearman $\rho$ = {sp:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=AXIS_FS,
            color=TREND_COLOR,
        )

    ax.set_xlabel("Instability score", fontsize=AXIS_FS)
    ax.set_ylabel("Ranking error", fontsize=AXIS_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_d_rejection(ax: plt.Axes, rejection_csv: str) -> None:
    rej = pd.read_csv(rejection_csv, comment="#")
    mapping = {"baseline": 0.0, "reject_10": 10.0, "reject_20": 20.0, "reject_30": 30.0}
    rows = []
    for s, pct in mapping.items():
        row = rej.loc[rej["setting"] == s, "mean_error_all"]
        if not row.empty:
            rows.append((pct, float(row.iloc[0])))
    rows.sort(key=lambda t: t[0])
    xs = np.array([t[0] for t in rows])
    ys = np.array([t[1] for t in rows])

    ax.plot(xs, ys, marker="o", linewidth=1.3, color="0.2", markersize=5)
    ax.set_xlabel("Rejection rate (%)", fontsize=AXIS_FS)
    ax.set_ylabel("Mean ranking error", fontsize=AXIS_FS)
    ax.set_xticks(xs)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    for px, py in zip(xs, ys):
        ax.annotate(
            f"{py:.4f}",
            (px, py),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=ANNOT_FS - 1,
            clip_on=True,
        )
    ax.margins(y=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_d_seed_risk_coverage(
    ax: plt.Axes,
    *,
    killer_json_path: str,
    search_root: str,
    proxy_key: str = "u6",
) -> None:
    """Risk–coverage for two seeds (same data as seed_killer phase3 left panel)."""
    with open(killer_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    slug = str(meta["slug"])
    dataset = str(meta["dataset"])
    category = str(meta["category"])
    kshot = int(meta["k"])
    sa, sb = int(meta["seed_a"]), int(meta["seed_b"])
    proxy_col = PROXY_COL.get(proxy_key, PROXY_COL["u6"])
    cov = default_coverage_grid()

    def _curve(seed: int) -> pd.DataFrame:
        job = os.path.join(search_root, slug, str(seed))
        csv_path = find_per_sample_csv(job, dataset, kshot, seed, category)
        if not csv_path:
            raise FileNotFoundError(
                f"Panel (d): no per_sample CSV for slug={slug} seed={seed} under {job}"
            )
        df_raw = load_and_validate_csv(csv_path)
        df = build_analysis_frame(df_raw, _LOG_PANEL_D)
        return risk_coverage(df, proxy_col, cov)

    ca = _curve(sa)
    cb = _curve(sb)
    colors = ("#1f77b4", "#ff7f0e")
    for c, seed, color in ((ca, sa, colors[0]), (cb, sb, colors[1])):
        xs = c["coverage"].to_numpy(dtype=float)
        ys = c["mean_error"].to_numpy(dtype=float)
        order = np.argsort(xs, kind="mergesort")
        ax.plot(xs[order], ys[order], linewidth=1.5, color=color, label=f"seed {seed}")

    ax.set_xlabel("Coverage", fontsize=AXIS_FS)
    ax.set_ylabel("Mean ranking error", fontsize=AXIS_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        list(reversed(handles)),
        list(reversed(labels)),
        loc="upper right",
        bbox_to_anchor=(1.0, 1.05),
        borderaxespad=0,
        frameon=False,
        fontsize=TICK_FS,
    )
    ax.set_xlim(float(np.min(cov)), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=str, default="PromptAD/result_round1")
    parser.add_argument(
        "--exp_summary_csv",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--macaroni_exp_dir",
        type=str,
        default=(
            "PromptAD/result_round1/visa/k_2/pairwise_instability/"
            "CLS-visa-macaroni1-k2-seed111-per_sample/experiments"
        ),
    )
    parser.add_argument(
        "--killer_pair_json",
        type=str,
        default="PromptAD/result_seed_search/analysis/killer_pair.json",
        help="Metadata + seeds for panel (d) risk–coverage curves.",
    )
    parser.add_argument(
        "--seed_search_root",
        type=str,
        default="PromptAD/result_seed_search",
        help="Root containing {slug}/{seed}/... per-sample CSVs.",
    )
    parser.add_argument(
        "--seed_proxy",
        type=str,
        default="u6",
        choices=sorted(PROXY_COL.keys()),
        help="Proxy column for ordering rejects (same as seed killer phase3).",
    )
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(_repo_root(), args.result_root))
    exp_csv = os.path.abspath(
        os.path.join(
            _repo_root(),
            args.exp_summary_csv or os.path.join(args.result_root, "exp_summary_all.csv"),
        )
    )
    mac_dir = os.path.abspath(os.path.join(_repo_root(), args.macaroni_exp_dir))
    fig_dir = os.path.join(root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = load_enriched_table(exp_csv, root)

    sample_csv = os.path.join(mac_dir, "exp5_sample_ranking_error.csv")
    killer_json = os.path.abspath(os.path.join(_repo_root(), args.killer_pair_json))
    search_root = os.path.abspath(os.path.join(_repo_root(), args.seed_search_root))

    # --- Standalone panel (a) ---
    fig_a, ax_a = plt.subplots(figsize=(4.8, 3.9))
    draw_panel_a_counterexample(ax_a, df)
    ax_a.set_title("AUROC vs instability", fontsize=TITLE_FS)
    fig_a.tight_layout()
    p_a = os.path.join(fig_dir, "figure_a_counterexample.png")
    fig_a.savefig(p_a, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig_a)

    # --- 2x2 final: larger bottom-right cell so panel (d) matches visual weight of (a–c) ---
    fig = plt.figure(figsize=(8.9, 6.35))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.0, 1.32],
        height_ratios=[1.0, 1.14],
        left=0.11,
        right=0.98,
        top=0.92,
        bottom=0.10,
        wspace=0.26,
        hspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    titles = (
        "AUROC vs instability",
        "Distribution of instability",
        "Instability vs error",
        "Same AUROC, different decision reliability",
    )

    draw_panel_a_counterexample(ax_a, df)
    ax_a.set_title(titles[0], fontsize=TITLE_FS)

    draw_panel_b_flip_hist(ax_b, df)
    ax_b.set_title(titles[1], fontsize=TITLE_FS)

    draw_panel_c_instability_error(ax_c, sample_csv)
    ax_c.set_title(titles[2], fontsize=TITLE_FS)

    draw_panel_d_seed_risk_coverage(
        ax_d,
        killer_json_path=killer_json,
        search_root=search_root,
        proxy_key=args.seed_proxy,
    )
    ax_d.set_title(titles[3], fontsize=TITLE_FS)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), ("a", "b", "c", "d")):
        ax.text(
            -0.12,
            1.05,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=PANEL_LETTER_FS,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="0.1",
            zorder=10,
            clip_on=False,
        )

    out_png = os.path.join(fig_dir, "Analysis_PromptAD_final.png")
    out_pdf = os.path.join(fig_dir, "Analysis_PromptAD_final.pdf")
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("Wrote:", p_a)
    print("Wrote:", out_png)
    print("Wrote:", out_pdf)


if __name__ == "__main__":
    main()
