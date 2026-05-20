#!/usr/bin/env python3
"""
Paper-style Figure 1 for Sec 3.1.1 PromptAD (cached tables only).

Presentation-oriented layout: full-grid strict pairwise evidence from aggregates only.
Does NOT read pairwise_metrics.csv, unified raw, or rerun PromptAD / pairwise recompute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

PREFERRED_SCOPE = ("mvtec", "toothbrush", 1)


def _setting_row(met: pd.DataFrame, d: str, c: str, shot: int, seed: int) -> Optional[pd.Series]:
    m = met[
        (met["dataset"].astype(str) == d)
        & (met["category"].astype(str) == c)
        & (met["shot"].astype(int) == int(shot))
        & (met["seed"].astype(int) == int(seed))
    ]
    if len(m) == 0:
        return None
    return m.iloc[0]


def _pair_from_met_rows(r_a: pd.Series, r_b: pd.Series) -> Dict[str, Any]:
    a_a = float(r_a["setting_auroc"])
    a_b = float(r_b["setting_auroc"])
    i_a = float(r_a["mean_pairwise_instability"])
    i_b = float(r_b["mean_pairwise_instability"])
    sa, sb = int(r_a["seed"]), int(r_b["seed"])
    if sa > sb:
        sa, sb = sb, sa
        r_a, r_b = r_b, r_a
        a_a, a_b = a_b, a_a
        i_a, i_b = i_b, i_a
    return {
        "dataset": str(r_a["dataset"]),
        "category": str(r_a["category"]),
        "shot": int(r_a["shot"]),
        "seed_a": sa,
        "seed_b": sb,
        "setting_auroc_a": a_a,
        "setting_auroc_b": a_b,
        "mean_pairwise_instability_a": i_a,
        "mean_pairwise_instability_b": i_b,
        "auroc_gap": float(abs(a_a - a_b)),
        "instability_gap": float(abs(i_a - i_b)),
    }


def _rc_for_setting(rc: pd.DataFrame, d: str, c: str, shot: int, seed: int) -> pd.DataFrame:
    sub = rc[
        (rc["dataset"].astype(str) == d)
        & (rc["category"].astype(str) == c)
        & (rc["shot"].astype(int) == int(shot))
        & (rc["seed"].astype(int) == int(seed))
    ].sort_values("coverage")
    return sub.copy()


def _smooth_rc_for_plot(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub[["coverage", "mean_pairwise_risk"]].copy()
    out["mean_pairwise_risk"] = pd.to_numeric(out["mean_pairwise_risk"], errors="coerce")
    out = out.sort_values("coverage")
    out["mean_pairwise_risk"] = out["mean_pairwise_risk"].ffill().bfill()
    return out


def _mean_abs_risk_diff(rc: pd.DataFrame, d: str, c: str, shot: int, s1: int, s2: int) -> float:
    rc1 = _smooth_rc_for_plot(_rc_for_setting(rc, d, c, shot, s1))
    rc2 = _smooth_rc_for_plot(_rc_for_setting(rc, d, c, shot, s2))
    if len(rc1) == 0 or len(rc2) == 0:
        return 0.0
    cov = np.union1d(rc1["coverage"].to_numpy(), rc2["coverage"].to_numpy())
    r1i = np.interp(cov, rc1["coverage"], rc1["mean_pairwise_risk"], left=np.nan, right=np.nan)
    r2i = np.interp(cov, rc2["coverage"], rc2["mean_pairwise_risk"], left=np.nan, right=np.nan)
    mask = np.isfinite(r1i) & np.isfinite(r2i)
    if not mask.any():
        return 0.0
    return float(np.nanmean(np.abs(r1i[mask] - r2i[mask])))


def _met_seed_pairs_for_group(met: pd.DataFrame, d: str, c: str, shot: int) -> List[Dict[str, Any]]:
    g = met[(met["dataset"].astype(str) == d) & (met["category"].astype(str) == c) & (met["shot"].astype(int) == shot)]
    if len(g) < 2:
        return []
    g = g.sort_values("seed")
    seeds = g["seed"].to_numpy(dtype=int)
    rows = {int(s): g[g["seed"] == s].iloc[0] for s in seeds}
    out: List[Dict[str, Any]] = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            out.append(_pair_from_met_rows(rows[int(seeds[i])], rows[int(seeds[j])]))
    return out


def _rank_pair_candidates(met: pd.DataFrame, same_au: pd.DataFrame, rc: pd.DataFrame) -> List[Dict[str, Any]]:
    """Enrich pairs with risk separation; prefer same_au + preferred-scope met pairs."""
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, int, int, int]] = set()

    def _key(p: Dict[str, Any]) -> Tuple[str, str, int, int, int]:
        return (p["dataset"], p["category"], int(p["shot"]), int(p["seed_a"]), int(p["seed_b"]))

    if len(same_au) > 0:
        ig = same_au["instability_gap"].astype(float)
        inst_hi = float(np.nanpercentile(ig, 90)) if np.isfinite(ig).any() else float("inf")
        inst_hi = max(inst_hi, 1e-6)

        for _, r in same_au.iterrows():
            d, c, shot = str(r["dataset"]), str(r["category"]), int(r["shot"])
            s1, s2 = int(r["seed_a"]), int(r["seed_b"])
            ra = _setting_row(met, d, c, shot, s1)
            rb = _setting_row(met, d, c, shot, s2)
            if ra is None or rb is None:
                continue
            p = _pair_from_met_rows(ra, rb)
            k = _key(p)
            if k in seen:
                continue
            seen.add(k)
            risk_sep = _mean_abs_risk_diff(rc, d, c, shot, p["seed_a"], p["seed_b"])
            inst_gap = float(p["instability_gap"])
            auroc_gap = float(p["auroc_gap"])
            outlier_pen = 0.45 if inst_gap > inst_hi * 1.35 else 1.0
            pref = 2.2 if (d, c, shot) == PREFERRED_SCOPE else 1.0
            score = pref * outlier_pen * (risk_sep + 1e-4) * np.log1p(inst_gap) / (auroc_gap + 5e-4)
            rows.append(
                {
                    "pair": p,
                    "source_row": "same_auroc_instability_pairs_csv",
                    "risk_mean_abs_diff": risk_sep,
                    "score": float(score),
                    "auroc_gap": auroc_gap,
                    "instability_gap": inst_gap,
                }
            )

    d0, c0, sh0 = PREFERRED_SCOPE
    for p in _met_seed_pairs_for_group(met, d0, c0, sh0):
        k = _key(p)
        if k in seen:
            continue
        seen.add(k)
        risk_sep = _mean_abs_risk_diff(rc, p["dataset"], p["category"], p["shot"], p["seed_a"], p["seed_b"])
        pref = 2.5
        inst_gap = float(p["instability_gap"])
        auroc_gap = float(p["auroc_gap"])
        if auroc_gap > 0.02:
            continue
        outlier_pen = 1.0
        score = pref * outlier_pen * (risk_sep + 1e-4) * np.log1p(max(inst_gap, 1e-6)) / (auroc_gap + 5e-4)
        rows.append(
            {
                "pair": p,
                "source_row": "setting_level_mvtec_toothbrush_k1_all_seed_pairs",
                "risk_mean_abs_diff": risk_sep,
                "score": float(score),
                "auroc_gap": auroc_gap,
                "instability_gap": inst_gap,
            }
        )

    if not rows:
        for (d, c, shot), _g in met.groupby(["dataset", "category", "shot"], sort=False):
            for p in _met_seed_pairs_for_group(met, str(d), str(c), int(shot)):
                k = _key(p)
                if k in seen:
                    continue
                seen.add(k)
                risk_sep = _mean_abs_risk_diff(rc, p["dataset"], p["category"], p["shot"], p["seed_a"], p["seed_b"])
                rows.append(
                    {
                        "pair": p,
                        "source_row": "setting_level_auto_group",
                        "risk_mean_abs_diff": risk_sep,
                        "score": float((risk_sep + 1e-4) * np.log1p(p["instability_gap"]) / (p["auroc_gap"] + 5e-4)),
                        "auroc_gap": float(p["auroc_gap"]),
                        "instability_gap": float(p["instability_gap"]),
                    }
                )

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def _choose_pair_for_figure(
    ranked: List[Dict[str, Any]],
    *,
    min_risk_sep_for_panel_d: float,
    max_try: int,
) -> Tuple[Dict[str, Any], str, bool, List[Dict[str, Any]]]:
    """
    Pick best pair; if top choices have very weak risk–curve separation, advance down the list.
    Returns (pair_dict, selection_reason, fallback_global, tried_notes).
    """
    tried: List[Dict[str, Any]] = []
    if not ranked:
        raise SystemExit("no pair candidates for paper-style Fig1")

    for i, row in enumerate(ranked[:max_try]):
        p = row["pair"]
        rs = float(row["risk_mean_abs_diff"])
        tried.append({"rank": i, "risk_mean_abs_diff": rs, "source_row": row["source_row"], "pair": p})
        weak = rs < min_risk_sep_for_panel_d and i < max_try - 1
        if weak:
            continue
        reason = row["source_row"]
        if i > 0:
            reason += f";advanced_from_rank0_weak_risk(risk_sep<{min_risk_sep_for_panel_d})"
        fb = row["source_row"] == "setting_level_auto_group"
        return p, reason, fb, tried

    p = ranked[0]["pair"]
    return p, ranked[0]["source_row"] + ";best_effort_weak_risk_separation", True, tried


def _panel_letter(ax: Any, letter: str) -> None:
    ax.text(
        0.02,
        0.98,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        va="top",
        ha="left",
        color="0.15",
        zorder=20,
    )


def _per_seed_metrics_for_summary(
    met: pd.DataFrame, d: str, c: str, shot: int, s1: int, s2: int
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for lab, sd in (("seed_a", s1), ("seed_b", s2)):
        r = _setting_row(met, d, c, shot, sd)
        if r is None:
            out[lab] = None
            continue
        block: Dict[str, Any] = {
            "setting_auroc": float(r["setting_auroc"]),
            "mean_pairwise_instability": float(r["mean_pairwise_instability"]),
        }
        if "mean_pairwise_error" in r.index:
            block["mean_pairwise_error"] = float(r["mean_pairwise_error"])
        out[lab] = block
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build paper-style Sec3 PromptAD Figure 1 from cached CSVs.")
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/cached_results/sec3_promptad"))
    ap.add_argument("--fig-dir", type=Path, default=Path("outputs/figures/sec3_promptad"))
    ap.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/cached_results/sec3_promptad/paper_style_fig1_summary.json"),
    )
    ap.add_argument(
        "--near-auroc-eps",
        type=float,
        default=0.01,
        help="Reference for summary JSON (caption-scale near AUROC).",
    )
    ap.add_argument(
        "--min-risk-sep-panel-d",
        type=float,
        default=0.0035,
        help="If top-ranked pair has mean abs risk-curve diff below this, try the next candidate.",
    )
    args = ap.parse_args()

    cache_dir = args.cache_dir.resolve()
    fig_dir = args.fig_dir.resolve()
    summary_path = args.summary_out.resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    met_p = cache_dir / "setting_level_metrics.csv"
    sau_p = cache_dir / "same_auroc_instability_pairs.csv"
    rc_p = cache_dir / "risk_coverage.csv"
    for p in (met_p, rc_p):
        if not p.is_file():
            raise SystemExit(f"missing required cache file: {p}")
    same_au = pd.read_csv(sau_p) if sau_p.is_file() else pd.DataFrame()

    met = pd.read_csv(met_p)
    rc = pd.read_csv(rc_p)

    need_m = {
        "dataset",
        "category",
        "shot",
        "seed",
        "setting_auroc",
        "mean_pairwise_instability",
        "mean_pairwise_error",
    }
    if not need_m.issubset(met.columns):
        raise SystemExit(f"setting_level_metrics.csv missing columns: {sorted(need_m - set(met.columns))}")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )

    ranked = _rank_pair_candidates(met, same_au, rc)
    pair_sel, sel_reason, fallback_used, tried = _choose_pair_for_figure(
        ranked,
        min_risk_sep_for_panel_d=args.min_risk_sep_panel_d,
        max_try=12,
    )
    risk_sep_sel = _mean_abs_risk_diff(
        rc,
        pair_sel["dataset"],
        pair_sel["category"],
        int(pair_sel["shot"]),
        int(pair_sel["seed_a"]),
        int(pair_sel["seed_b"]),
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.4), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- (a) AUROC vs instability ---
    ax_a.scatter(
        met["setting_auroc"],
        met["mean_pairwise_instability"],
        s=22,
        alpha=0.38,
        c="0.35",
        edgecolors="none",
        rasterized=True,
    )
    ra = _setting_row(met, pair_sel["dataset"], pair_sel["category"], pair_sel["shot"], pair_sel["seed_a"])
    rb = _setting_row(met, pair_sel["dataset"], pair_sel["category"], pair_sel["shot"], pair_sel["seed_b"])
    if ra is not None and rb is not None:
        xa = [float(ra["setting_auroc"]), float(rb["setting_auroc"])]
        ya = [float(ra["mean_pairwise_instability"]), float(rb["mean_pairwise_instability"])]
        ax_a.plot(xa, ya, color="0.75", lw=1.4, zorder=5, linestyle="-")
        ax_a.scatter(
            [xa[0]],
            [ya[0]],
            s=85,
            c="#1f77b4",
            edgecolors="white",
            linewidths=0.6,
            zorder=8,
        )
        ax_a.scatter(
            [xa[1]],
            [ya[1]],
            s=85,
            c="#d62728",
            edgecolors="white",
            linewidths=0.6,
            zorder=8,
        )
        d_auc = abs(xa[0] - xa[1])
        d_i = abs(ya[0] - ya[1])
        cat = str(ra["category"])
        ann = f"{cat}\nd instability = {d_i:.4f}\nd AUROC = {d_auc:.4f}"
        ax_a.annotate(
            ann,
            xy=((xa[0] + xa[1]) / 2, (ya[0] + ya[1]) / 2),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.92),
            arrowprops=dict(arrowstyle="-|>", color="0.55", lw=0.9, relpos=(0.5, 0.0)),
        )
    ax_a.set_xlabel("Setting AUROC")
    ax_a.set_ylabel("Mean pairwise instability")
    ax_a.set_title("AUROC vs instability", fontsize=9.5, pad=4)
    _panel_letter(ax_a, "a")
    ax_a.grid(alpha=0.22)

    # --- (b) distribution ---
    inst = met["mean_pairwise_instability"].astype(float).to_numpy()
    inst = inst[np.isfinite(inst)]
    mean_inst = float(np.mean(inst)) if inst.size else float("nan")
    n_set = len(inst)
    n_bins = int(np.clip(round(1.5 * np.sqrt(max(n_set, 1))), 18, 36))
    ax_b.hist(inst, bins=n_bins, color="0.45", edgecolor="white", linewidth=0.4, alpha=0.88)
    ax_b.axvline(mean_inst, color="darkorange", lw=1.8, label="Mean", zorder=6)
    ymax_hist = float(ax_b.get_ylim()[1])
    if inst.size >= 8:
        try:
            kde = gaussian_kde(inst)
            xs = np.linspace(float(np.min(inst)), float(np.max(inst)), 160)
            dens = kde(xs)
            dens = dens / (np.max(dens) + 1e-12) * (ymax_hist * 0.24 + 1e-6)
            ax_b.plot(xs, dens, color="firebrick", lw=1.15, alpha=0.75, label="KDE (shape)")
        except Exception:
            pass
    ax_b.set_xlabel("Mean pairwise instability")
    ax_b.set_ylabel("Number of settings")
    ax_b.set_title("Distribution of instability", fontsize=9.5, pad=4)
    _panel_letter(ax_b, "b")
    ax_b.legend(loc="upper right", framealpha=0.9)
    ax_b.grid(axis="y", alpha=0.22)

    # --- (c) setting-level instability vs error ---
    mx = met["mean_pairwise_instability"].astype(float)
    my = met["mean_pairwise_error"].astype(float)
    mask = np.isfinite(mx) & np.isfinite(my)
    mxv = mx[mask].to_numpy()
    myv = my[mask].to_numpy()
    rho_val: Optional[float] = None
    rho_p: Optional[float] = None
    if len(mxv) >= 3:
        rho, pval = spearmanr(mxv, myv)
        rho_val = float(rho) if np.isfinite(rho) else None
        rho_p = float(pval) if np.isfinite(pval) else None

    ax_c.scatter(mxv, myv, s=26, alpha=0.5, c="0.25", edgecolors="none", rasterized=True)
    if len(mxv) >= 3:
        coef = np.polyfit(mxv, myv, 1)
        xs = np.linspace(np.min(mxv), np.max(mxv), 80)
        ax_c.plot(xs, np.poly1d(coef)(xs), color="darkred", lw=1.2, alpha=0.85, label="OLS fit")
    ax_c.set_xlabel("Mean pairwise instability")
    ax_c.set_ylabel("Mean pairwise error")
    ax_c.set_title("Instability vs ranking error", fontsize=9.5, pad=4)
    _panel_letter(ax_c, "c")
    if rho_val is not None:
        ax_c.text(
            0.04,
            0.96,
            f"ρ = {rho_val:.3f}",
            transform=ax_c.transAxes,
            va="top",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9),
        )
    ax_c.legend(loc="lower right", framealpha=0.9)
    ax_c.grid(alpha=0.22)

    # --- (d) risk–coverage ---
    d, c, shot = pair_sel["dataset"], pair_sel["category"], int(pair_sel["shot"])
    s1, s2 = int(pair_sel["seed_a"]), int(pair_sel["seed_b"])
    rc1 = _smooth_rc_for_plot(_rc_for_setting(rc, d, c, shot, s1))
    rc2 = _smooth_rc_for_plot(_rc_for_setting(rc, d, c, shot, s2))
    risk_diff_summary: Dict[str, Any] = {}
    if len(rc1) and len(rc2):
        cov = np.union1d(rc1["coverage"].to_numpy(), rc2["coverage"].to_numpy())
        r1i = np.interp(cov, rc1["coverage"], rc1["mean_pairwise_risk"], left=np.nan, right=np.nan)
        r2i = np.interp(cov, rc2["coverage"], rc2["mean_pairwise_risk"], left=np.nan, right=np.nan)
        m = np.isfinite(r1i) & np.isfinite(r2i)
        if m.any():
            c_m = cov[m]
            j = int(np.argmax(c_m))
            risk_diff_summary = {
                "mean_abs_risk_diff_interpolated": float(np.nanmean(np.abs(r1i[m] - r2i[m]))),
                "mean_pairwise_risk_at_max_coverage_seed_a": float(r1i[m][j]),
                "mean_pairwise_risk_at_max_coverage_seed_b": float(r2i[m][j]),
            }
        ax_d.plot(rc1["coverage"], rc1["mean_pairwise_risk"], color="#1f77b4", lw=1.8)
        ax_d.plot(rc2["coverage"], rc2["mean_pairwise_risk"], color="#d62728", lw=1.8)

    per = _per_seed_metrics_for_summary(met, d, c, shot, s1, s2)
    la = per.get("seed_a") or {}
    lb = per.get("seed_b") or {}

    def _leg_txt(sd: int, blk: Dict[str, Any]) -> str:
        if not blk:
            return f"seed {sd}"
        au = blk.get("setting_auroc", float("nan"))
        ins = blk.get("mean_pairwise_instability", float("nan"))
        return f"seed {sd}\nAUROC={au:.3f}\nI={ins:.3f}"

    leg_handles = [
        mlines.Line2D([], [], color="#1f77b4", lw=1.8),
        mlines.Line2D([], [], color="#d62728", lw=1.8),
    ]
    ax_d.legend(
        leg_handles,
        [_leg_txt(s1, la), _leg_txt(s2, lb)],
        loc="lower right",
        framealpha=0.94,
        handlelength=1.8,
        borderpad=0.35,
    )
    ax_d.set_xlim(0, 1)
    ax_d.set_xlabel("Coverage")
    ax_d.set_ylabel("Mean pairwise error")
    ax_d.set_title("Reliability vs coverage", fontsize=9.5, pad=4)
    _panel_letter(ax_d, "d")
    ax_d.grid(alpha=0.22)

    png = fig_dir / "paper_style_fig1.png"
    pdf = fig_dir / "paper_style_fig1.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    selected_pair = {
        "dataset": pair_sel["dataset"],
        "category": pair_sel["category"],
        "shot": int(pair_sel["shot"]),
        "seed_a": int(pair_sel["seed_a"]),
        "seed_b": int(pair_sel["seed_b"]),
        "setting_auroc_a": float(pair_sel["setting_auroc_a"]),
        "setting_auroc_b": float(pair_sel["setting_auroc_b"]),
        "mean_pairwise_instability_a": float(pair_sel["mean_pairwise_instability_a"]),
        "mean_pairwise_instability_b": float(pair_sel["mean_pairwise_instability_b"]),
        "auroc_gap": float(pair_sel["auroc_gap"]),
        "instability_gap": float(pair_sel["instability_gap"]),
    }

    summary: Dict[str, Any] = {
        "definition": "strict_pairwise_cached_aggregates_only",
        "inputs": {
            "setting_level_metrics": str(met_p),
            "same_auroc_instability_pairs": str(sau_p) if sau_p.is_file() else None,
            "risk_coverage": str(rc_p),
        },
        "selected_pair": selected_pair,
        "selection_reason": sel_reason,
        "fallback_used": bool(fallback_used),
        "pair_risk_metrics": {
            "mean_abs_risk_curve_diff": float(risk_sep_sel),
            **risk_diff_summary,
        },
        "pair_auroc_instability": {
            "auroc_gap": float(pair_sel["auroc_gap"]),
            "instability_gap": float(pair_sel["instability_gap"]),
            "per_seed": _per_seed_metrics_for_summary(met, d, c, shot, s1, s2),
        },
        "ranking": {"top_candidates_tried": tried[:8]},
        "parameters": {
            "near_auroc_eps_reference": args.near_auroc_eps,
            "min_risk_sep_panel_d": args.min_risk_sep_panel_d,
        },
        "panel_b": {"n_settings": int(len(met)), "mean_setting_instability": mean_inst},
        "panel_c": {
            "mode": "setting_level_one_point_per_setting",
            "n_settings": int(len(met)),
            "spearman_rho": rho_val,
            "spearman_p": rho_p,
        },
        "outputs": {"png": str(png), "pdf": str(pdf), "summary_json": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
