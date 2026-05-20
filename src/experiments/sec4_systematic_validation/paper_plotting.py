#!/usr/bin/env python3
"""Matplotlib renderers for Section 4 paper figures (data from sec4_near_auroc_universe)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.sec4_systematic_validation.sec4_near_auroc_universe import (
    BUCKET_LABELS,
    BUCKET_ORDER,
    DEFAULT_COVERAGES_FIG7,
    SIGNAL_ORDER,
    build_risk_lookup,
)

REGIME_LABELS = ["Low", "Mid", "High"]


def paper_rc() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_png_pdf(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _reindex_margin(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["margin_bucket"] = d["margin_bucket"].astype(str).str.lower()
    return d.set_index("margin_bucket").reindex(BUCKET_ORDER).reset_index()


def plot_fig2_same_auroc(
    selection: pd.Series,
    roles: Dict[str, Any],
    setting_df: pd.DataFrame,
    rc_df: pd.DataFrame,
    out_base: Path,
) -> None:
    from src.experiments.sec4_systematic_validation.sec4_near_auroc_universe import _risk_at_coverage

    paper_rc()
    rc_lookup = build_risk_lookup(rc_df)
    cov = float(selection["coverage"])
    eps = float(selection["epsilon"])
    d, cat, shot = str(selection["dataset"]), str(selection["category"]), int(selection["shot"])
    g = setting_df[
        (setting_df["dataset"] == d)
        & (setting_df["category"] == cat)
        & (setting_df["shot"] == shot)
    ]
    xs, ys, cs, seeds = [], [], [], []
    for _, r in g.iterrows():
        sd = int(r["seed"])
        rc = rc_lookup.get((d, cat, shot, sd))
        if rc is None:
            continue
        risk = _risk_at_coverage(rc, cov)
        if not np.isfinite(risk):
            continue
        xs.append(float(r["setting_auroc"]))
        ys.append(risk)
        cs.append(float(r["mean_pairwise_instability"]))
        seeds.append(sd)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=70, edgecolors="k", linewidths=0.5)
    for x, y, sd in zip(xs, ys, seeds):
        ax.text(x + 1e-4, y + 1e-4, str(sd), fontsize=8)

    def _mark(seed: int, marker: str, label: str, color: str) -> None:
        rr = g[g["seed"] == seed].iloc[0]
        rc = rc_lookup.get((d, cat, shot, seed))
        yv = _risk_at_coverage(rc, cov)
        ax.scatter(
            [float(rr["setting_auroc"])],
            [yv],
            marker=marker,
            s=144,
            facecolors="none",
            edgecolors=color,
            linewidths=2,
            label=label,
        )

    _mark(int(roles["seed_auc"]), "s", "AUROC-selected", "red")
    _mark(int(roles["seed_inst"]), "D", "Instability-selected", "blue")
    _mark(int(roles["seed_oracle"]), "*", "Oracle", "orange")
    fig.colorbar(sc, ax=ax).set_label("mean_pairwise_instability")
    ax.set_xlabel("AUROC")
    ax.set_ylabel(f"Decision risk @ coverage={cov:.1f}")
    ax.set_title("Same AUROC, different decision risk", fontsize=12, pad=10)
    ax.text(
        0.02,
        0.98,
        f"{d} / {cat} / k={shot} (epsilon={eps:.3f})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.35",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", markerscale=0.8)
    fig.tight_layout()
    save_png_pdf(fig, out_base)


def plot_fig3_4_merged(chain: pd.DataFrame, cm: pd.DataFrame, y_fail: np.ndarray, out_base: Path) -> None:
    mc = _reindex_margin(chain)
    cmt = _reindex_margin(cm)
    x = np.arange(3, dtype=np.float64)
    y_inst = mc["mean_instability"].to_numpy(dtype=float)
    y_err_s = cmt["error_low_I"].to_numpy(dtype=float)
    y_err_u = cmt["error_high_I"].to_numpy(dtype=float)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=False)
    lw, ms = 2.0, 5

    ax = axes[0]
    ax.plot(x, y_inst, "o-", color="#C44E52", lw=lw, ms=ms)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Margin bucket")
    ax.set_ylabel("Mean instability")
    ax.set_title("(a) Instability")
    ax.grid(True, alpha=0.35, linewidth=0.5)
    _despine(ax)

    ax = axes[1]
    ax.plot(x, y_err_s, "o-", color="#1f77b4", lw=lw, ms=ms, label=r"Stable ($I_{\mathrm{flip}}=0$)")
    ax.plot(x, y_err_u, "o-", color="#ff7f0e", lw=lw, ms=ms, label=r"Unstable ($I_{\mathrm{flip}}=1$)")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Margin bucket")
    ax.set_ylabel("Mean pairwise error")
    ax.set_title("(b) Ranking error")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.35, linewidth=0.5)
    _despine(ax)

    ax = axes[2]
    ax.plot(x, y_fail, "o-", color="#2ca02c", lw=lw, ms=ms)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS)
    ax.set_xlabel("Instability bucket (AUROC-seed instability)")
    ax.set_ylabel("Failure rate")
    ax.set_title("(c) Failure rate")
    if np.any(np.isfinite(y_fail)):
        ymin = float(np.nanmin(y_fail))
        ymax = float(np.nanmax(y_fail))
        span = max(ymax - ymin, 1e-6)
        ax.set_ylim(max(0.0, ymin - 0.05 * span), min(1.0, ymax + 0.12 * span))
    ax.grid(True, alpha=0.35, linewidth=0.5)
    _despine(ax)
    fig.tight_layout()
    save_png_pdf(fig, out_base)


def plot_fig5_6_merged(fc: pd.DataFrame, y_regime: np.ndarray, out_base: Path) -> None:
    g = fc.set_index("signal").reindex(SIGNAL_ORDER).reset_index()
    y_fail = g["mean_failure"].astype(float).to_numpy()
    y_nf = g["mean_non_failure"].astype(float).to_numpy()

    plt.rcParams.update(
        {
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.5))
    x = np.arange(len(SIGNAL_ORDER))
    bar_w = 0.36
    ax_a.bar(x - bar_w / 2, y_fail, width=bar_w, color="#1f77b4", label="Failure")
    ax_a.bar(x + bar_w / 2, y_nf, width=bar_w, color="#ff7f0e", label="Non-failure")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(SIGNAL_ORDER)
    ax_a.set_xlabel("Signal type (AUROC-selected seed)")
    ax_a.set_ylabel("Mean value")
    ax_a.set_title("(a) Signal comparison")
    ax_a.legend(frameon=False, loc="best")
    ax_a.grid(True, axis="y", alpha=0.35)
    _despine(ax_a)

    xb = np.arange(3)
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.42 + 0.22 * i) for i in range(3)]
    ax_b.bar(xb, y_regime, width=bar_w, color=colors, edgecolor="none")
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(REGIME_LABELS)
    ax_b.set_xlabel("Instability regime")
    ax_b.set_ylabel("Failure rate")
    ax_b.set_title("(b) Failure rate vs instability")
    if np.any(np.isfinite(y_regime)):
        ymax = float(np.nanmax(y_regime))
        ax_b.set_ylim(0.0, min(1.0, ymax * 1.25 + 0.05))
    ax_b.grid(True, axis="y", alpha=0.35)
    _despine(ax_b)
    fig.tight_layout(pad=0.6, w_pad=2.0)
    save_png_pdf(fig, out_base)


def plot_fig7_aggregate_delta_risk(agg_df: pd.DataFrame, out_base: Path) -> None:
    paper_rc()
    fig, ax = plt.subplots(figsize=(7, 5))
    coverages_sorted = list(DEFAULT_COVERAGES_FIG7)[::-1]
    eps_sorted = sorted(agg_df["epsilon"].unique().tolist())
    x = np.arange(len(coverages_sorted), dtype=np.float64)
    width = 0.35 if len(eps_sorted) == 2 else max(0.8 / max(1, len(eps_sorted)), 0.2)
    y_all: list[float] = []
    for i, eps in enumerate(eps_sorted):
        ys = []
        sub = agg_df[agg_df["epsilon"] == eps]
        for c in coverages_sorted:
            r = sub[np.isclose(sub["coverage"].astype(float), float(c), rtol=0, atol=1e-9)]
            ys.append(float(r["mean_delta_risk"].iloc[0]) if not r.empty else float("nan"))
        y_all.extend(ys)
        shift = (i - (len(eps_sorted) - 1) / 2.0) * width
        ax.bar(x + shift, ys, width=width, label=f"epsilon={eps:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.1f}" for c in coverages_sorted])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("mean_delta_risk (risk_AUROC − risk_instability)")
    ax.set_title("Aggregate Delta Risk (near-AUROC pair universe)")
    ax.axhline(0.0, color="black", linewidth=1)
    y_arr = np.array([v for v in y_all if np.isfinite(v)], dtype=float)
    if y_arr.size:
        ymax = float(np.max(np.abs(y_arr)))
        y_half = max(ymax * 1.25, 0.01)
        ax.set_ylim(-y_half * 0.15, y_half)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    save_png_pdf(fig, out_base)
