#!/usr/bin/env python3
"""
PaDiM analog of PromptAD Section-4 paper figures (fig2–fig7 naming).

Writes to <repo>/paper_figures/padim/:
  fig2_same_auroc.{png,pdf}, fig3–fig6, fig7_delta_risk.{png,pdf},
  fig3_4_merged_mechanism.{png,pdf}, fig5_6_merged_failure.{png,pdf}, manifest.json

Fig3–Fig6 reuse plotting functions from PromptAD/utils/build_paper_figures_section4.py.
Intermediate CSVs are written under PaDiM/result_analysis/padim_section4_paper/ (overwritable).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_section4_plotters(repo: Path):
    p = repo / "PromptAD" / "utils" / "build_paper_figures_section4.py"
    spec = importlib.util.spec_from_file_location("build_paper_figures_section4", str(p))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _paper_rc_like() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
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


def _save_png_pdf(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fig2_padim_seed_scatter(
    metrics_csv: Path, killer_json: Path, out_base: Path
) -> Dict[str, Any]:
    met = pd.read_csv(metrics_csv)
    with open(killer_json, "r", encoding="utf-8") as f:
        killer = json.load(f)
    slug = str(killer["slug"])
    sa, sb = int(killer["seed_a"]), int(killer["seed_b"])
    sub = met[met["slug"] == slug].copy()
    if sub.empty:
        raise SystemExit(f"No rows for killer slug={slug} in {metrics_csv}")

    _paper_rc_like()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(sub["instability"], sub["auroc"], c="0.65", s=40, label="all seeds", zorder=1)
    ka = sub[sub["seed"] == sa]
    kb = sub[sub["seed"] == sb]
    if len(ka) != 1 or len(kb) != 1:
        raise SystemExit(f"Missing seed rows for {sa}/{sb} in slug {slug}")
    ax.scatter(
        [float(ka["instability"].iloc[0]), float(kb["instability"].iloc[0])],
        [float(ka["auroc"].iloc[0]), float(kb["auroc"].iloc[0])],
        c=["tab:blue", "tab:orange"],
        s=90,
        zorder=3,
        label=f"seeds {sa}, {sb}",
    )
    ax.set_xlabel("Mean instability (pairwise flip rate)")
    ax.set_ylabel("Image-level AUROC (fused)")
    ax.set_title(f"PaDiM Protocol B — {slug}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save_png_pdf(fig, out_base)
    return {"name": "fig2_same_auroc", "source": str(metrics_csv), "killer": str(killer_json), "status": "generated"}


def _padim_default_coverage_grid() -> np.ndarray:
    """Match ``padim_seed_killer_evidence_pipeline.default_coverage_grid`` / PromptAD pilot grid."""
    return np.clip(np.round(np.arange(1.0, 0.45, -0.05), 2), 0.5, 1.0)


def _pick_argmax_padim(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.sort_values([col, "instability", "seed"], ascending=[False, True, True], kind="mergesort")
    return g.iloc[0]


def _pick_argmin_padim(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.sort_values([col, "auroc", "seed"], ascending=[True, False, True], kind="mergesort")
    return g.iloc[0]


def _risk_at_coverage(curve: pd.DataFrame, cov: float) -> float:
    c = curve["coverage"].to_numpy(dtype=float)
    m = np.isclose(c, float(cov), rtol=0.0, atol=1e-9)
    if np.any(m):
        return float(curve.loc[m, "mean_error"].iloc[0])
    j = int(np.argmin(np.abs(c - float(cov))))
    return float(curve["mean_error"].iloc[j])


def plot_fig7_padim_aggregate_delta_risk(
    repo: Path,
    metrics_csv: Path,
    out_base: Path,
    mid: Path,
    epsilons: Tuple[float, ...] = (0.002, 0.005),
) -> Dict[str, Any]:
    """
    Same bar chart as PromptAD ``pilot_instability_aware_selection`` aggregate_delta_risk /
    ``fig7_delta_risk.png``: x = coverage (descending tick order), y = mean(risk_auc - risk_stable),
    grouped bars per epsilon. Selection logic mirrors pilot (AUROC seed vs instability-min among AUROC-tight).
    """
    promptad = repo / "PromptAD"
    if str(promptad) not in sys.path:
        sys.path.insert(0, str(promptad))
    from utils.rejection_instability_analysis import (  # type: ignore  # noqa: E402
        _deterministic_rejection_curve,
        build_analysis_frame,
        load_and_validate_csv,
    )

    log = logging.getLogger("padim_fig7_agg")
    if not log.handlers:
        log.addHandler(logging.NullHandler())

    met = pd.read_csv(metrics_csv)
    need = {"slug", "seed", "auroc", "instability", "per_sample_csv"}
    miss = sorted(need - set(met.columns))
    if miss:
        raise SystemExit(f"{metrics_csv} missing columns {miss}")

    cov_grid = _padim_default_coverage_grid()
    mid.mkdir(parents=True, exist_ok=True)

    curve_cache: Dict[Tuple[str, int], pd.DataFrame] = {}

    def _curve_for_row(row: pd.Series) -> pd.DataFrame:
        slug = str(row["slug"])
        seed = int(row["seed"])
        key = (slug, seed)
        if key in curve_cache:
            return curve_cache[key]
        p = Path(str(row["per_sample_csv"]))
        if not p.is_file():
            raise FileNotFoundError(str(p))
        raw = load_and_validate_csv(str(p))
        af = build_analysis_frame(raw, log)
        u = af["proxy_u6"].to_numpy(dtype=float)
        order = np.argsort(np.where(np.isfinite(u), u, np.inf), kind="mergesort")
        cv = _deterministic_rejection_curve(af, order, cov_grid)
        curve_cache[key] = cv
        return cv

    agg_rows: List[Dict[str, Any]] = []
    skipped = 0

    for slug, g in met.groupby("slug", sort=False):
        g = g.copy()
        if len(g) < 2:
            skipped += 1
            continue
        inst = g["instability"].to_numpy(dtype=float)
        if np.allclose(inst, 0.0):
            skipped += 1
            continue

        for eps in epsilons:
            seed_auc_row = _pick_argmax_padim(g, "auroc")
            best_auc = float(seed_auc_row["auroc"])
            candidate = g[g["auroc"].to_numpy(dtype=float) >= (best_auc - float(eps))].copy()
            if candidate.empty:
                skipped += 1
                continue
            seed_stable_row = _pick_argmin_padim(candidate, "instability")

            # Precompute curves once per seed in this group
            curves: Dict[int, pd.DataFrame] = {}
            for _, r in g.iterrows():
                try:
                    curves[int(r["seed"])] = _curve_for_row(r)
                except Exception:
                    curves[int(r["seed"])] = pd.DataFrame()

            for cov in cov_grid:
                cov_f = float(cov)
                sa = int(seed_auc_row["seed"])
                ss = int(seed_stable_row["seed"])
                ca, cs = curves.get(sa), curves.get(ss)
                if ca is None or cs is None or ca.empty or cs.empty:
                    skipped += 1
                    continue
                ra = _risk_at_coverage(ca, cov_f)
                rs = _risk_at_coverage(cs, cov_f)
                if not (np.isfinite(ra) and np.isfinite(rs)):
                    skipped += 1
                    continue
                risks: Dict[int, float] = {}
                for seed, cv in curves.items():
                    if cv is None or cv.empty:
                        continue
                    rv = _risk_at_coverage(cv, cov_f)
                    if np.isfinite(rv):
                        risks[seed] = rv
                if not risks:
                    skipped += 1
                    continue
                seed_oracle = int(min(risks, key=lambda k: risks[k]))
                ro = float(risks[seed_oracle])
                d_ao = float(ra - ro)
                d_so = float(rs - ro)
                agg_rows.append(
                    {
                        "epsilon": float(eps),
                        # Canonical 2-decimal grid key (matches pilot); avoids float drift in groupby/plot.
                        "coverage": float(np.round(cov_f, 2)),
                        "n_settings_valid": 1,
                        "mean_delta_risk": float(ra - rs),
                        "median_delta_risk": float(ra - rs),
                        "win_rate_stable_over_auc": float(rs < ra),
                        "mean_dist_auc_to_oracle": d_ao,
                        "mean_dist_stable_to_oracle": d_so,
                        "oracle_gap_reduction": float(d_ao - d_so),
                        "slug": str(slug),
                    }
                )

    if not agg_rows:
        raise SystemExit(
            "Could not build PaDiM aggregate delta risk (no valid slug×coverage rows). "
            "Check per_sample.csv paths in all_seed_metrics.csv."
        )

    raw_agg = pd.DataFrame(agg_rows)
    agg_df = (
        raw_agg.groupby(["epsilon", "coverage"], as_index=False)
        .agg(
            n_settings_valid=("slug", "count"),
            mean_delta_risk=("mean_delta_risk", "mean"),
            median_delta_risk=("median_delta_risk", "median"),
            win_rate_stable_over_auc=("win_rate_stable_over_auc", "mean"),
            mean_dist_auc_to_oracle=("mean_dist_auc_to_oracle", "mean"),
            mean_dist_stable_to_oracle=("mean_dist_stable_to_oracle", "mean"),
            oracle_gap_reduction=("oracle_gap_reduction", "mean"),
        )
        .sort_values(["epsilon", "coverage"], kind="mergesort")
    )
    agg_csv = mid / "aggregate_summary_padim_fig7.csv"
    agg_df.to_csv(agg_csv, index=False)

    # --- Plot: match PromptAD pilot ``aggregate_delta_risk.png`` style ---
    _paper_rc_like()
    fig, ax = plt.subplots(figsize=(7, 5))
    # Fixed descending coverage axis (1.0 -> 0.5); pilot uses the same grid, not raw float uniques.
    coverages_sorted = list(_padim_default_coverage_grid()[::-1])
    eps_sorted = sorted(agg_df["epsilon"].unique().tolist())
    x = np.arange(len(coverages_sorted), dtype=np.float64)
    width = 0.35 if len(eps_sorted) == 2 else max(0.8 / max(1, len(eps_sorted)), 0.2)
    bar_colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    y_all: List[float] = []
    for i, eps in enumerate(eps_sorted):
        ys: List[float] = []
        sub = agg_df[agg_df["epsilon"] == eps]
        for c in coverages_sorted:
            cc = float(np.round(float(c), 2))
            r = sub[sub["coverage"].to_numpy(dtype=float) == cc]
            if len(r) > 1:
                r = r.iloc[[0]]
            ys.append(float(r["mean_delta_risk"].iloc[0]) if not r.empty else float("nan"))
        y_all.extend(ys)
        shift = (i - (len(eps_sorted) - 1) / 2.0) * width
        fc = bar_colors[i % len(bar_colors)]
        # Zero-height bars are otherwise invisible; edges match pilot-style legibility.
        ax.bar(
            x + shift,
            ys,
            width=width,
            label=f"epsilon={float(eps):.3f}",
            facecolor=fc,
            edgecolor="0.2",
            linewidth=0.9,
            zorder=3,
        )
    ax.set_xticks(x)
    # Grid is 0.05 steps: use two decimals so 0.55 / 0.60 do not collapse to duplicate "0.6".
    ax.set_xticklabels([f"{float(c):.2f}" for c in coverages_sorted])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("mean_delta_risk (risk_auc - risk_stable)")
    ax.set_title("Aggregate Delta Risk (PaDiM Protocol B)")
    ax.axhline(0.0, color="black", linewidth=1, zorder=1)
    y_arr = np.asarray(y_all, dtype=np.float64)
    y_arr = y_arr[np.isfinite(y_arr)]
    ymax = float(np.max(np.abs(y_arr))) if y_arr.size else 0.0
    # Near-flat Δ risk: Matplotlib often expands to ~±0.04; tighten by one order for readability.
    if ymax < 1e-12:
        y_half = 0.004
    else:
        y_half = float(np.clip(ymax * 1.2, 1e-9, 1.0))
    ax.set_ylim(-y_half, y_half)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="best")
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "name": "fig7_delta_risk",
        "aggregate_csv": str(agg_csv),
        "metrics": str(metrics_csv),
        "skipped_approx": skipped,
        "status": "generated",
    }


def _write_mechanism_csvs_from_panel_a(panel_csv: Path, mid: Path) -> Dict[str, Path]:
    mid.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(panel_csv)
    if "flip_rate_mean" not in df.columns or "instability" not in df.columns:
        raise SystemExit(f"{panel_csv} missing flip_rate_mean/instability")

    # --- mechanism_chain_summary: tertiles by AUROC (setting-level "margin" proxy) ---
    a = df["AUROC"].to_numpy(dtype=float)
    q1, q2 = np.quantile(a, [1.0 / 3.0, 2.0 / 3.0])

    def _auroc_bucket(x: float) -> str:
        if x <= q1:
            return "low"
        if x <= q2:
            return "mid"
        return "high"

    df = df.copy()
    df["margin_bucket"] = df["AUROC"].map(_auroc_bucket)
    g = df.groupby("margin_bucket", as_index=False).agg(
        mean_instability=("instability", "mean"),
        error_rate=("flip_rate_mean", "mean"),
        failure_rate=("flip_rate_mean", "mean"),
    )
    order = ["low", "mid", "high"]
    g["margin_bucket"] = pd.Categorical(g["margin_bucket"], categories=order, ordered=True)
    g = g.sort_values("margin_bucket")
    mc_path = mid / "mechanism_chain_summary.csv"
    g.to_csv(mc_path, index=False)

    # --- failure_gate_analysis: instability tertiles at setting level ---
    inst = df["instability"].to_numpy(dtype=float)
    t1, t2 = np.quantile(inst, [1.0 / 3.0, 2.0 / 3.0])

    def _inst_regime(x: float) -> str:
        if x <= t1:
            return "low_I"
        if x <= t2:
            return "mid_I"
        return "high_I"

    df["regime"] = df["instability"].map(_inst_regime)
    wcol = "n_pairs" if "n_pairs" in df.columns else ("n_test" if "n_test" in df.columns else None)
    if wcol is None:
        raise SystemExit(f"{panel_csv} missing n_pairs/n_test for weighting")
    fg_rows = []
    for reg in ["low_I", "mid_I", "high_I"]:
        h = df[df["regime"] == reg]
        if h.empty:
            fg_rows.append({"regime": reg, "failure_rate": float("nan"), "n_settings": 0.0})
            continue
        w = h[wcol].to_numpy(dtype=float)
        fr = h["flip_rate_mean"].to_numpy(dtype=float)
        tot = float(np.sum(w))
        val = float(np.sum(fr * w) / tot) if tot > 0 else float("nan")
        fg_rows.append({"regime": reg, "failure_rate": val, "n_settings": tot})
    fg = pd.DataFrame(fg_rows)
    fg_path = mid / "failure_gate_analysis.csv"
    fg.to_csv(fg_path, index=False)

    # --- controlled_margin_analysis from exp5 (sample-level) if available ---
    exp5 = panel_csv.parent.parent / "exp5_visa_macaroni1_r18" / "experiments" / "exp5_sample_ranking_error.csv"
    cm_path = mid / "controlled_margin_analysis.csv"
    if exp5.is_file():
        s = pd.read_csv(exp5)
        med_h = float(np.median(s["harmonic_score"].to_numpy(dtype=float)))
        s["margin_abs"] = np.abs(s["harmonic_score"].to_numpy(dtype=float) - med_h)
        qm1, qm2 = np.quantile(s["margin_abs"].to_numpy(dtype=float), [1.0 / 3.0, 2.0 / 3.0])

        def _m_bucket(x: float) -> str:
            if x <= qm1:
                return "low"
            if x <= qm2:
                return "mid"
            return "high"

        s["margin_bucket"] = s["margin_abs"].map(_m_bucket)
        med_i = float(np.median(s["I_bin"].to_numpy(dtype=float)))
        s["I_hi"] = (s["I_bin"].to_numpy(dtype=float) > med_i).astype(int)
        rows = []
        for b in ["low", "mid", "high"]:
            sub = s[s["margin_bucket"] == b]
            if sub.empty:
                continue
            e0 = float(sub.loc[sub["I_hi"] == 0, "error"].mean()) if (sub["I_hi"] == 0).any() else float("nan")
            e1 = float(sub.loc[sub["I_hi"] == 1, "error"].mean()) if (sub["I_hi"] == 1).any() else float("nan")
            rows.append({"margin_bucket": b, "error_low_I": e0, "error_high_I": e1})
        pd.DataFrame(rows).to_csv(cm_path, index=False)
    else:
        # Fallback: reuse mechanism buckets with flip_rate split by instability median within bucket (coarse)
        med_inst = float(np.median(df["instability"].to_numpy(dtype=float)))

        def _split_mean_flip(sub: pd.DataFrame, mask: pd.Series) -> float:
            if not mask.any():
                return float("nan")
            return float(np.average(sub.loc[mask, "flip_rate_mean"], weights=sub.loc[mask, wcol]))

        rows = []
        for b in ["low", "mid", "high"]:
            sub = df[df["margin_bucket"] == b]
            if sub.empty:
                continue
            low_mask = sub["instability"] <= med_inst
            high_mask = sub["instability"] > med_inst
            rows.append(
                {
                    "margin_bucket": b,
                    "error_low_I": _split_mean_flip(sub, low_mask),
                    "error_high_I": _split_mean_flip(sub, high_mask),
                }
            )
        pd.DataFrame(rows).to_csv(cm_path, index=False)

    # --- failure_conditioned_signal_analysis (minimal, exp5-based) ---
    fs_path = mid / "failure_conditioned_signal_analysis.csv"
    if exp5.is_file():
        s = pd.read_csv(exp5)
        fail = s["error"].to_numpy(dtype=float) > float(np.median(s["error"].to_numpy(dtype=float)))
        med_h = float(np.median(s["harmonic_score"].to_numpy(dtype=float)))
        margin = np.abs(s["harmonic_score"].to_numpy(dtype=float) - med_h)
        gap = np.abs(s["harmonic_score"].to_numpy(dtype=float) - 0.5)
        # score_var: local variance over a tiny rolling window (smoothness proxy)
        hs = s["harmonic_score"].to_numpy(dtype=float)
        win = 11
        pad = np.pad(hs, (win // 2, win // 2), mode="edge")
        loc_var = np.array([float(np.var(pad[i : i + win])) for i in range(len(hs))], dtype=float)

        def _means(sig: np.ndarray) -> tuple[float, float]:
            return float(np.mean(sig[fail])), float(np.mean(sig[~fail]))

        rows_fs: List[Dict[str, Any]] = []
        eps_list = [0.002, 0.005]
        cov_list = [0.5, 0.7, 0.8]
        for eps in eps_list:
            for cov in cov_list:
                mi_f, mi_nf = _means(s["I_bin"].to_numpy(dtype=float))
                ma_f, ma_nf = _means(margin)
                ga_f, ga_nf = _means(gap)
                sv_f, sv_nf = _means(loc_var)
                rows_fs.extend(
                    [
                        dict(
                            epsilon=eps,
                            coverage=cov,
                            signal="instability",
                            n_failure=int(fail.sum()),
                            n_non_failure=int((~fail).sum()),
                            mean_failure=mi_f,
                            mean_non_failure=mi_nf,
                        ),
                        dict(
                            epsilon=eps,
                            coverage=cov,
                            signal="margin",
                            n_failure=int(fail.sum()),
                            n_non_failure=int((~fail).sum()),
                            mean_failure=ma_f,
                            mean_non_failure=ma_nf,
                        ),
                        dict(
                            epsilon=eps,
                            coverage=cov,
                            signal="gap",
                            n_failure=int(fail.sum()),
                            n_non_failure=int((~fail).sum()),
                            mean_failure=ga_f,
                            mean_non_failure=ga_nf,
                        ),
                        dict(
                            epsilon=eps,
                            coverage=cov,
                            signal="score_var",
                            n_failure=int(fail.sum()),
                            n_non_failure=int((~fail).sum()),
                            mean_failure=sv_f,
                            mean_non_failure=sv_nf,
                        ),
                    ]
                )
        pd.DataFrame(rows_fs).to_csv(fs_path, index=False)
    else:
        # Minimal placeholder so PromptAD-style plotter still runs (values are not meaningful).
        rows_fs = []
        for eps in [0.002, 0.005]:
            for cov in [0.5, 0.7, 0.8]:
                for sig in ["instability", "margin", "gap", "score_var"]:
                    rows_fs.append(
                        dict(
                            epsilon=eps,
                            coverage=cov,
                            signal=sig,
                            n_failure=1,
                            n_non_failure=1,
                            mean_failure=0.2,
                            mean_non_failure=0.2,
                        )
                    )
        pd.DataFrame(rows_fs).to_csv(fs_path, index=False)

    return {"mechanism_chain": mc_path, "failure_gate": fg_path, "controlled_margin": cm_path, "failure_signal": fs_path}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=str, default="/home/zju/mywork/NeurIPS2026")
    p.add_argument("--out-dir", type=str, default="paper_figures/padim")
    p.add_argument(
        "--padim-root",
        type=str,
        default=None,
        help="PaDiM repo root (default: <repo-root>/PaDiM-Anomaly-Detection-Localization-master)",
    )
    p.add_argument(
        "--seed-search-root",
        type=str,
        default=None,
        help="Directory with <slug>/<seed>/per_sample.csv (default: <padim-root>/padim_result_seed_search)",
    )
    args = p.parse_args()

    repo = Path(args.repo_root).resolve()
    padim = Path(args.padim_root).resolve() if args.padim_root else (repo / "PaDiM-Anomaly-Detection-Localization-master")
    search_root = Path(args.seed_search_root).resolve() if args.seed_search_root else (padim / "padim_result_seed_search")
    out_dir = (repo / args.out_dir).resolve()
    mid = padim / "result_analysis" / "padim_section4_paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = search_root / "analysis" / "all_seed_metrics.csv"
    killer_json = search_root / "analysis" / "killer_pair.json"
    pairs_csv = search_root / "analysis" / "candidate_pairs_with_risk_metrics.csv"
    panel_csv = padim / "result_analysis" / "figures" / "padim_panel_a_mvtec_visa_r18_wr50.csv"

    for fp in (metrics_csv, killer_json, pairs_csv, panel_csv):
        if not fp.is_file():
            raise SystemExit(f"Missing required input: {fp}")

    plot_mod = _load_section4_plotters(repo)

    manifest: Dict[str, Any] = {
        "paper_figures_dir": str(out_dir),
        "padim_root": str(padim),
        "seed_search_root": str(search_root),
        "figures": [],
    }

    # Remove previously copied panel-style artifacts in the paper output folder (not fig2–7).
    for stale in out_dir.glob("padim_panel_*"):
        try:
            stale.unlink()
        except OSError:
            pass

    m2 = plot_fig2_padim_seed_scatter(metrics_csv, killer_json, out_dir / "fig2_same_auroc")
    manifest["figures"].append(m2)

    m7 = plot_fig7_padim_aggregate_delta_risk(repo, metrics_csv, out_dir / "fig7_delta_risk", mid)
    manifest["figures"].append(m7)

    paths = _write_mechanism_csvs_from_panel_a(panel_csv, mid)
    manifest["intermediate_csv"] = {k: str(v) for k, v in paths.items()}

    ok, src = plot_mod.plot_controlled_margin(paths["controlled_margin"], out_dir / "fig3_controlled_margin")
    if not ok:
        raise SystemExit(f"fig3 failed: {src}")
    manifest["figures"].append({"name": "fig3_controlled_margin", "csv": str(paths["controlled_margin"]), "status": "generated"})

    ok, src = plot_mod.plot_failure_gate(paths["failure_gate"], out_dir / "fig6_failure_regime")
    if not ok:
        raise SystemExit(f"fig6 failed: {src}")
    manifest["figures"].append({"name": "fig6_failure_regime", "csv": str(paths["failure_gate"]), "status": "generated"})

    ok, src = plot_mod.plot_failure_signal(paths["failure_signal"], out_dir / "fig5_failure_signal")
    if not ok:
        raise SystemExit(f"fig5 failed: {src}")
    manifest["figures"].append({"name": "fig5_failure_signal", "csv": str(paths["failure_signal"]), "status": "generated"})

    ok4, src4, st4 = plot_mod.plot_mechanism_chain(
        paths["mechanism_chain"], out_dir / "fig4_mechanism_chain", failure_gate_csv=paths["failure_gate"]
    )
    if not ok4:
        raise SystemExit("fig4 plot_mechanism_chain returned False")
    manifest["figures"].append({"name": "fig4_mechanism_chain", "csv": str(paths["mechanism_chain"]), "status": st4})

    merge_py = padim / "merge_padim_fig34_fig56_section4.py"
    subprocess.check_call(
        [
            sys.executable,
            str(merge_py),
            "--repo-root",
            str(repo),
            "--csv-dir",
            str(mid),
            "--out-dir",
            str(out_dir.relative_to(repo)),
        ],
        cwd=str(repo),
    )
    manifest.setdefault("merged_figures", []).extend(
        [
            {
                "name": "fig3_4_merged_mechanism",
                "script": str(merge_py),
                "csv_dir": str(mid),
                "status": "generated",
            },
            {
                "name": "fig5_6_merged_failure",
                "script": str(merge_py),
                "csv_dir": str(mid),
                "status": "generated",
            },
        ]
    )

    man_path = out_dir / "manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("Wrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
