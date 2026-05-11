"""
Figure styling for PaDiM panels (b)(c)(d): verbatim from PromptAD
`utils/build_analysis_promptad_final.py` (fonts, colors, layout).

Panel (d) risk–coverage uses the same computation as ``padim_seed_killer_evidence_pipeline`` phase3
(left subplot of ``killer_final.png``): ``killer_pair.json`` + ``per_sample.csv`` per seed.
"""
from __future__ import annotations

import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from padim_seed_killer_evidence_pipeline import (
    PROXY_COL as _SEED_KILLER_PROXY_COL,
    build_analysis_frame_for_proxy,
    default_coverage_grid,
    find_per_sample_csv,
    risk_coverage,
)

_LOG_PANEL_D = logging.getLogger("padim_figure_promptad_style.panel_d_seed")
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


def draw_panel_d_seed_risk_coverage(
    ax: plt.Axes,
    *,
    killer_json_path: str,
    search_root: str,
    proxy_key: str = "u6",
) -> None:
    """Risk–coverage for two seeds (native redraw of killer_final.png left panel)."""
    with open(killer_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    slug = str(meta["slug"])
    sa, sb = int(meta["seed_a"]), int(meta["seed_b"])
    proxy = proxy_key if proxy_key in _SEED_KILLER_PROXY_COL else "u6"
    proxy_col = _SEED_KILLER_PROXY_COL[proxy]
    cov = default_coverage_grid()

    def _curve(seed: int) -> pd.DataFrame:
        job = os.path.join(search_root, slug, str(seed))
        csv_path = find_per_sample_csv(job)
        if not csv_path:
            raise FileNotFoundError(f"Panel (d): no per_sample.csv under {job}")
        df = build_analysis_frame_for_proxy(csv_path, proxy, _LOG_PANEL_D)
        if proxy_col not in df.columns:
            raise ValueError(f"{csv_path}: missing {proxy_col} after analysis frame build")
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
