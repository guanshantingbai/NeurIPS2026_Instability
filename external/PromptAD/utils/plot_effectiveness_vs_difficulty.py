#!/usr/bin/env python3
"""
Scatter: effectiveness of instability-based rejection vs per-setting difficulty
(mean sample_error from per_sample_instability_analysis.csv).

Supports u2 (legacy filenames) and u6 (margin-aware proxy). Does not modify any CSV.
matplotlib only (no seaborn).
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# User paths (edit as needed)
# ---------------------------------------------------------------------------

input_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "result_round1",
    "rejection_instability_aggregate",
    "proxy_comparison_all_settings.csv",
)

# Spec: ".PromptAD/scatter_plots/" — relative to NeurIPS2026 repo root (parent of PromptAD/)
_PROMPTAD_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_REPO_ROOT = os.path.dirname(_PROMPTAD_ROOT)
output_dir = os.path.join(_REPO_ROOT, ".PromptAD", "scatter_plots")

RESULT_ROOT = os.path.normpath(os.path.join(os.path.dirname(input_csv), ".."))

# (proxy_name, file_inst_vs_random, file_inst_vs_score)
SCATTER_SPECS = [
    ("u2", "scatter_inst_vs_random.png", "scatter_inst_vs_score.png"),
    ("u6", "scatter_u6_vs_random.png", "scatter_u6_vs_score.png"),
]


def _region_label_x(lo: float, hi: float, xmin: float, xmax: float) -> float:
    """Horizontal center of [lo, hi] clipped to current x-axis; robust if interval is empty."""
    a = max(lo, xmin)
    b = min(hi, xmax)
    if b > a:
        return float(0.5 * (a + b))
    return float(0.5 * (xmin + xmax))


def instability_analysis_csv(result_root: str, setting_csv: str) -> str:
    """
    setting_csv is relative to result_root, e.g. mvtec/k_1/csv/CLS-...-per_sample.csv
    -> .../mvtec/k_1/rejection_instability/<stem>/per_sample_instability_analysis.csv
    """
    stem = os.path.splitext(os.path.basename(setting_csv))[0]
    full_input = os.path.join(result_root, setting_csv)
    k_dir = os.path.dirname(os.path.dirname(full_input))
    return os.path.join(k_dir, "rejection_instability", stem, "per_sample_instability_analysis.csv")


def load_mean_sample_error(path: str) -> float:
    d = pd.read_csv(path)
    if "sample_error" not in d.columns:
        raise ValueError("no column sample_error")
    return float(d["sample_error"].mean())


def load_proxy_scatter_arrays(
    df_all: pd.DataFrame,
    proxy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Returns (x_arr, y1_arr, y2_arr) for one proxy, or None if no valid points.
    y2 = mean_delta_risk_inst_vs_score (instability vs score-based rejection).
    """
    df = df_all[df_all["proxy_name"] == proxy].copy()
    if df.empty:
        return None

    xs: list[float] = []
    y1: list[float] = []
    y2: list[float] = []

    for _, row in df.iterrows():
        setting_csv = str(row["setting_csv"])
        apath = instability_analysis_csv(RESULT_ROOT, setting_csv)
        try:
            mx = load_mean_sample_error(apath)
        except FileNotFoundError:
            warnings.warn(f"missing file, skip: {apath}")
            continue
        except Exception as e:
            warnings.warn(f"read failed ({e}), skip: {apath}")
            continue

        xs.append(mx)
        y1.append(float(row["mean_delta_risk"]))
        y2.append(float(row["mean_delta_risk_inst_vs_score"]))

    if len(xs) < 1:
        return None

    return (
        np.asarray(xs, dtype=float),
        np.asarray(y1, dtype=float),
        np.asarray(y2, dtype=float),
    )


def plot_proxy_compare_u6_u2_vs_score(df_all: pd.DataFrame) -> None:
    """1x2 panel: u6 vs score (left), u2 vs score (right); shared axes; difficulty regimes."""
    loaded_u6 = load_proxy_scatter_arrays(df_all, "u6")
    loaded_u2 = load_proxy_scatter_arrays(df_all, "u2")
    if loaded_u6 is None or loaded_u2 is None:
        warnings.warn("plot_proxy_compare_u6_u2_vs_score: missing u6 or u2 data; skip combined figure.")
        return

    x_u6, _, y_u6 = loaded_u6
    x_u2, _, y_u2 = loaded_u2

    x_all = np.concatenate([x_u6, x_u2])
    y_all = np.concatenate([y_u6, y_u2])
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    pad_x = 0.02 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    panels: list[tuple[plt.Axes, np.ndarray, np.ndarray, str]] = [
        (axes[0], x_u6, y_u6, "u6 (margin-aware)"),
        (axes[1], x_u2, y_u2, "u2 (baseline)"),
    ]

    ylabel = "ΔRisk (Instability - Score-based)"
    xlabel = "Mean Sample Error (per setting)"

    for ax, x_arr, y_arr, subtitle in panels:
        ax.scatter(x_arr, y_arr, alpha=0.75, s=36, edgecolors="k", linewidths=0.3)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.axvline(0.05, color="gray", linestyle="--", alpha=0.6, linewidth=1.0, zorder=0)
        ax.axvline(0.15, color="gray", linestyle="--", alpha=0.6, linewidth=1.0, zorder=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(subtitle)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(ylabel)

    fig.tight_layout()

    for ax in axes:
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        y_text = ymin + 0.9 * (ymax - ymin)
        ax.text(
            _region_label_x(xmin, 0.05, xmin, xmax),
            y_text,
            "easy",
            ha="center",
            va="center",
            fontsize=11,
            zorder=5,
        )
        ax.text(
            _region_label_x(0.05, 0.15, xmin, xmax),
            y_text,
            "medium",
            ha="center",
            va="center",
            fontsize=11,
            zorder=5,
        )
        ax.text(
            _region_label_x(0.15, xmax, xmin, xmax),
            y_text,
            "hard",
            ha="center",
            va="center",
            fontsize=11,
            zorder=5,
        )

    out_path = os.path.join(output_dir, "proxy_compare_u6_u2_vs_score.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_one_proxy(
    df_all: pd.DataFrame,
    proxy: str,
    fname_random: str,
    fname_score: str,
) -> None:
    loaded = load_proxy_scatter_arrays(df_all, proxy)
    if loaded is None:
        warnings.warn(f"No valid points for proxy_name={proxy!r} after loading per-setting files; skip.")
        return

    x_arr, y1_arr, y2_arr = loaded

    os.makedirs(output_dir, exist_ok=True)

    def _scatter(
        y: np.ndarray,
        ylabel: str,
        fname: str,
        title: str,
        *,
        difficulty_regions: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x_arr, y, alpha=0.75, s=36, edgecolors="k", linewidths=0.3)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        if difficulty_regions:
            ax.axvline(
                0.05,
                color="gray",
                linestyle="--",
                alpha=0.6,
                linewidth=1.0,
                zorder=0,
            )
            ax.axvline(
                0.15,
                color="gray",
                linestyle="--",
                alpha=0.6,
                linewidth=1.0,
                zorder=0,
            )
        ax.set_xlabel("Mean Sample Error (per setting)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if difficulty_regions:
            ymin, ymax = ax.get_ylim()
            xmin, xmax = ax.get_xlim()
            y_text = ymin + 0.9 * (ymax - ymin)
            ax.text(
                _region_label_x(xmin, 0.05, xmin, xmax),
                y_text,
                "easy",
                ha="center",
                va="center",
                fontsize=11,
                zorder=5,
            )
            ax.text(
                _region_label_x(0.05, 0.15, xmin, xmax),
                y_text,
                "medium",
                ha="center",
                va="center",
                fontsize=11,
                zorder=5,
            )
            ax.text(
                _region_label_x(0.15, xmax, xmin, xmax),
                y_text,
                "hard",
                ha="center",
                va="center",
                fontsize=11,
                zorder=5,
            )
        out_path = os.path.join(output_dir, fname)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")

    _scatter(
        y1_arr,
        "ΔRisk (Instability - Random)",
        fname_random,
        f"Effectiveness vs difficulty ({proxy}, inst vs random)",
    )
    _scatter(
        y2_arr,
        "ΔRisk (Instability - Score-based)",
        fname_score,
        f"Effectiveness vs difficulty ({proxy}, inst vs score)",
    )
    if proxy == "u2":
        _scatter(
            y2_arr,
            "ΔRisk (Instability - Score-based)",
            "scatter_inst_vs_score_with_regions.png",
            "Instability vs. score-based rejection across difficulty regimes",
            difficulty_regions=True,
        )


def main() -> int:
    if not os.path.isfile(input_csv):
        print(f"ERROR: input CSV not found: {input_csv}", file=sys.stderr)
        return 1

    df_all = pd.read_csv(input_csv)
    need = [
        "setting_csv",
        "proxy_name",
        "mean_delta_risk",
        "mean_delta_risk_inst_vs_score",
    ]
    miss = [c for c in need if c not in df_all.columns]
    if miss:
        print(f"ERROR: missing columns: {miss}", file=sys.stderr)
        return 1

    for proxy, f1, f2 in SCATTER_SPECS:
        plot_one_proxy(df_all, proxy, f1, f2)

    plot_proxy_compare_u6_u2_vs_score(df_all)

    return 0


if __name__ == "__main__":
    sys.exit(main())
