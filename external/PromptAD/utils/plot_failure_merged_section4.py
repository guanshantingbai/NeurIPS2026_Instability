#!/usr/bin/env python3
"""
Merged failure figure (Section 4): signal separation + regime concentration.
Redraws from CSV only (no PDF concatenation).

Outputs: paper_figures/promptad/fig5_6_merged_failure.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIGNAL_ORDER = ["instability", "gap", "score_var", "margin"]
COVERAGES_FIG5 = [0.5, 0.7, 0.8]
REGIME_ORDER = ["low_I", "mid_I", "high_I"]
REGIME_LABELS = ["Low", "Mid", "High"]


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _aggregate_signal_comparison(fc: pd.DataFrame) -> pd.DataFrame:
    """Same logic as build_paper_figures_section4.plot_failure_signal."""
    df = fc.copy()
    df = df[df["signal"].isin(SIGNAL_ORDER)]
    df = df[df["coverage"].isin(COVERAGES_FIG5)]
    if df.empty:
        raise ValueError("No rows after filtering failure_conditioned_signal_analysis.csv")
    g = df.groupby("signal", as_index=False)[["mean_failure", "mean_non_failure"]].mean()
    g["signal"] = pd.Categorical(g["signal"], categories=SIGNAL_ORDER, ordered=True)
    return g.sort_values("signal").reset_index(drop=True)


def _weighted_failure_by_regime(fg: pd.DataFrame) -> np.ndarray:
    """Same weighting as build_paper_figures_section4.plot_failure_gate."""
    out = np.full(3, np.nan, dtype=np.float64)
    for i, reg in enumerate(REGIME_ORDER):
        sub = fg[fg["regime"].astype(str) == reg]
        if sub.empty:
            continue
        w = sub["n_settings"].to_numpy(dtype=np.float64)
        fr = sub["failure_rate"].to_numpy(dtype=np.float64)
        tot = float(np.sum(w))
        out[i] = float(np.sum(fr * w) / tot) if tot > 0 else np.nan
    return out


def plot_fig5_6_merged(
    failure_signal_csv: Path,
    failure_gate_csv: Path,
    out_pdf: Path,
    out_png: Path,
) -> None:
    fc = pd.read_csv(failure_signal_csv)
    fg = pd.read_csv(failure_gate_csv)

    sig_df = _aggregate_signal_comparison(fc)
    y_fail = sig_df["mean_failure"].to_numpy(dtype=np.float64)
    y_nf = sig_df["mean_non_failure"].to_numpy(dtype=np.float64)
    x_labels = sig_df["signal"].astype(str).tolist()

    y_regime = _weighted_failure_by_regime(fg)

    plt.rcParams.update(
        {
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.5))

    n_sig = len(x_labels)
    x = np.arange(n_sig, dtype=np.float64)
    bar_w = 0.36

    ax_a.bar(
        x - bar_w / 2,
        y_fail,
        width=bar_w,
        color="#1f77b4",
        label="Failure",
        edgecolor="none",
    )
    ax_a.bar(
        x + bar_w / 2,
        y_nf,
        width=bar_w,
        color="#ff7f0e",
        label="Non-failure",
        edgecolor="none",
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(x_labels)
    ax_a.set_xlabel("Signal type")
    ax_a.set_ylabel("Mean value")
    ax_a.set_title("(a) Signal comparison")
    ax_a.grid(True, axis="y", alpha=0.35, linestyle="-", linewidth=0.5)
    ax_a.legend(loc="best", frameon=False)
    _despine(ax_a)

    xb = np.arange(3, dtype=np.float64)
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.42 + 0.22 * i) for i in range(3)]
    ax_b.bar(xb, y_regime, width=bar_w, color=colors, edgecolor="none")
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(REGIME_LABELS)
    ax_b.set_xlabel("Instability regime")
    ax_b.set_ylabel("Failure rate")
    ax_b.set_title("(b) Failure rate vs instability")
    ax_b.grid(True, axis="y", alpha=0.35, linestyle="-", linewidth=0.5)
    _despine(ax_b)

    fig.tight_layout(pad=0.6, w_pad=2.0)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root containing PromptAD/ and paper_figures/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    strengthen = root / "PromptAD" / "result_analysis" / "promptad_strengthening"
    mech = root / "PromptAD" / "result_analysis" / "mechanism_analysis"
    fc_path = strengthen / "failure_conditioned_signal_analysis.csv"
    fg_path = mech / "failure_gate_analysis.csv"
    out_pdf = root / "paper_figures" / "promptad" / "fig5_6_merged_failure.pdf"
    out_png = root / "paper_figures" / "promptad" / "fig5_6_merged_failure.png"

    for p in (fc_path, fg_path):
        if not p.is_file():
            raise SystemExit(f"Missing required CSV: {p}")

    plot_fig5_6_merged(fc_path, fg_path, out_pdf, out_png)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
