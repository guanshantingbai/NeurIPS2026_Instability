#!/usr/bin/env python3
"""
Unified mechanism figure: margin → instability → error → failure (Section 4).
Redraws from CSV only (no PDF concatenation).

Outputs: paper_figures/promptad/fig3_4_merged_mechanism.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BUCKET_ORDER = ["low", "mid", "high"]
BUCKET_LABELS = ["Low", "Mid", "High"]
REGIME_ORDER = ["low_I", "mid_I", "high_I"]


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _reindex_margin(df: pd.DataFrame, col: str = "margin_bucket") -> pd.DataFrame:
    d = df.copy()
    d[col] = d[col].astype(str).str.lower()
    d = d.set_index(col).reindex(BUCKET_ORDER).reset_index()
    return d


def _weighted_failure_by_regime(fg: pd.DataFrame) -> np.ndarray:
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


def plot_mechanism_figure(
    chain_csv: Path,
    margin_csv: Path,
    gate_csv: Path,
    out_pdf: Path,
    out_png: Path,
) -> None:
    mc = _reindex_margin(pd.read_csv(chain_csv))
    cm = _reindex_margin(pd.read_csv(margin_csv))
    fg = pd.read_csv(gate_csv)

    x = np.arange(3, dtype=np.float64)
    y_inst = mc["mean_instability"].to_numpy(dtype=np.float64)
    y_err_stable = cm["error_low_I"].to_numpy(dtype=np.float64)
    y_err_unstable = cm["error_high_I"].to_numpy(dtype=np.float64)
    y_fail = _weighted_failure_by_regime(fg)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=False)
    lw = 2.0
    ms = 5

    # (a) Instability vs margin
    ax = axes[0]
    ax.plot(x, y_inst, marker="o", linewidth=lw, markersize=ms, color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Margin bucket")
    ax.set_ylabel("Mean instability")
    ax.set_title("(a) Instability")
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.5)
    _despine(ax)

    # (b) Error vs margin — two lines
    ax = axes[1]
    ax.plot(
        x,
        y_err_stable,
        marker="o",
        linewidth=lw,
        markersize=ms,
        color="#1f77b4",
        label=r"Stable ($I_{\mathrm{flip}}=0$)",
    )
    ax.plot(
        x,
        y_err_unstable,
        marker="o",
        linewidth=lw,
        markersize=ms,
        color="#ff7f0e",
        label=r"Unstable ($I_{\mathrm{flip}}=1$)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Margin bucket")
    ax.set_ylabel("Mean pairwise error")
    ax.set_title("(b) Ranking error")
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.5)
    ax.legend(loc="best", frameon=False)
    _despine(ax)

    # (c) Failure vs instability
    ax = axes[2]
    ax.plot(x, y_fail, marker="o", linewidth=lw, markersize=ms, color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Instability bucket")
    ax.set_ylabel("Failure rate")
    ax.set_title("(c) Failure rate")
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.5)
    _despine(ax)

    fig.tight_layout()
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
        help="NeurIPS2026 (or project) root containing PromptAD/ and paper_figures/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    mech = root / "PromptAD" / "result_analysis" / "mechanism_analysis"
    out_pdf = root / "paper_figures" / "promptad" / "fig3_4_merged_mechanism.pdf"
    out_png = root / "paper_figures" / "promptad" / "fig3_4_merged_mechanism.png"

    chain_csv = mech / "mechanism_chain_summary.csv"
    margin_csv = mech / "controlled_margin_analysis.csv"
    gate_csv = mech / "failure_gate_analysis.csv"
    for p in (chain_csv, margin_csv, gate_csv):
        if not p.is_file():
            raise SystemExit(f"Missing required CSV: {p}")

    plot_mechanism_figure(chain_csv, margin_csv, gate_csv, out_pdf, out_png)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
