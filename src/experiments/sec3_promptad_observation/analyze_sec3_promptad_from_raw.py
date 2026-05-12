#!/usr/bin/env python3
"""
Section 3.1.1 — empirical stats from PromptAD unified raw scores using the **pairwise** definition
(PromptAD-style anomaly–normal pairs), not a semantic–visual score-difference proxy.

No PromptAD training/inference. Reads ``unified_raw_scores_wide.csv`` (preferred) or pivots
``unified_raw_scores_long.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_wide_or_long(raw_dir: Path) -> pd.DataFrame:
    wide_p = raw_dir / "unified_raw_scores_wide.csv"
    long_p = raw_dir / "unified_raw_scores_long.csv"
    if wide_p.is_file():
        return pd.read_csv(wide_p)
    if not long_p.is_file():
        raise FileNotFoundError(f"missing {wide_p} and {long_p}")
    df = pd.read_csv(long_p)
    need = {"sample_id", "condition", "condition_score", "dataset", "category", "shot", "seed", "label", "image_path"}
    if not need.issubset(df.columns):
        raise ValueError(f"long CSV missing columns: {sorted(need - set(df.columns))}")
    piv = df.pivot_table(
        index=["sample_id", "dataset", "category", "shot", "seed", "label", "image_path"],
        columns="condition",
        values="condition_score",
        aggfunc="first",
    ).reset_index()
    for c in ("semantic", "visual", "fused"):
        if c not in piv.columns:
            raise ValueError(f"pivot missing condition column {c!r}")
    piv["semantic_score"] = piv["semantic"]
    piv["visual_score"] = piv["visual"]
    piv["fused_score"] = piv["fused"]
    if "config" in df.columns:
        cfg_map = df.drop_duplicates(subset=["sample_id"]).set_index("sample_id")["config"]
        piv["config"] = piv["sample_id"].map(cfg_map)
    else:
        piv["config"] = ""
    return piv


def _pairwise_instability(z_sem: float, z_vis: float) -> float:
    """Var([z_sem, z_vis]) with population variance (ddof=0), per paper-style pairwise branch disagreement."""
    return float(np.var([z_sem, z_vis], ddof=0))


def _build_pairs_for_setting(
    g: pd.DataFrame,
    *,
    max_pairs: Optional[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """All anomaly×normal pairs within one (dataset, category, shot, seed); optional uniform subsample."""
    pos = g[g["label"] == 1].reset_index(drop=True)
    neg = g[g["label"] == 0].reset_index(drop=True)
    if len(pos) == 0 or len(neg) == 0:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for i in range(len(pos)):
        for j in range(len(neg)):
            sa = float(pos.loc[i, "semantic_score"])
            sn = float(neg.loc[j, "semantic_score"])
            va = float(pos.loc[i, "visual_score"])
            vn = float(neg.loc[j, "visual_score"])
            fa = float(pos.loc[i, "fused_score"])
            fn = float(neg.loc[j, "fused_score"])
            z_sem = 1.0 if sa > sn else 0.0
            z_vis = 1.0 if va > vn else 0.0
            z_fused = 1.0 if fa > fn else 0.0
            p_inst = _pairwise_instability(z_sem, z_vis)
            p_err = 1.0 - z_fused
            margin = fa - fn
            rows.append(
                {
                    "dataset": pos.loc[i, "dataset"],
                    "category": pos.loc[i, "category"],
                    "shot": int(pos.loc[i, "shot"]),
                    "seed": int(pos.loc[i, "seed"]),
                    "image_path_anomaly": str(pos.loc[i, "image_path"]),
                    "image_path_normal": str(neg.loc[j, "image_path"]),
                    "sample_id_anomaly": str(pos.loc[i, "sample_id"]),
                    "sample_id_normal": str(neg.loc[j, "sample_id"]),
                    "z_sem": z_sem,
                    "z_vis": z_vis,
                    "z_fused": z_fused,
                    "pairwise_instability": p_inst,
                    "pairwise_error": p_err,
                    "fused_margin": margin,
                    "semantic_score_anomaly": sa,
                    "semantic_score_normal": sn,
                    "visual_score_anomaly": va,
                    "visual_score_normal": vn,
                    "fused_score_anomaly": fa,
                    "fused_score_normal": fn,
                }
            )

    out = pd.DataFrame(rows)
    if max_pairs is not None and len(out) > max_pairs:
        idx = rng.choice(len(out), size=max_pairs, replace=False)
        out = out.iloc[idx].reset_index(drop=True)
    return out


def _sample_instability_from_pairs(pair_df: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Per image: mean pairwise_instability and mean pairwise_error over pairs where it appears."""
    out = g[["dataset", "category", "shot", "seed", "sample_id", "image_path", "label"]].copy()
    out["image_path"] = out["image_path"].astype(str)
    mp_i: List[float] = []
    mp_e: List[float] = []
    n_p: List[int] = []
    roles: List[str] = []
    for _, row in out.iterrows():
        path = row["image_path"]
        lab = int(row["label"])
        if lab == 1:
            sub = pair_df[pair_df["image_path_anomaly"] == path]
            role = "anomaly"
        else:
            sub = pair_df[pair_df["image_path_normal"] == path]
            role = "normal"
        roles.append(role)
        if len(sub) == 0:
            mp_i.append(float("nan"))
            mp_e.append(float("nan"))
            n_p.append(0)
        else:
            mp_i.append(float(sub["pairwise_instability"].mean()))
            mp_e.append(float(sub["pairwise_error"].mean()))
            n_p.append(int(len(sub)))
    out["role_in_pairs"] = roles
    out["mean_pairwise_instability"] = mp_i
    out["mean_pairwise_error"] = mp_e
    out["n_pairs"] = n_p
    return out


def _risk_coverage_curve(
    g: pd.DataFrame,
    sample_stats: pd.DataFrame,
    *,
    risk_by: str,
    max_pairs: Optional[int],
    rng: np.random.Generator,
    n_grid: int,
) -> pd.DataFrame:
    """
    Grow an accepted subset in order of increasing _risk (low _risk first).
    instability: _risk = mean pairwise instability (NaNs last).
    fused: _risk = -fused_score so low _risk = high fused (accept high fused first).
    Within kept anomalies and normals, rebuild anomaly–normal pairs and report mean pairwise_error.
    """
    ss = sample_stats.copy()
    if risk_by == "instability":
        ss["_risk"] = ss["mean_pairwise_instability"].fillna(np.inf)
    elif risk_by == "fused":
        m = g.set_index("image_path")["fused_score"].astype(float)
        fused = ss["image_path"].map(m)
        ss["_risk"] = -fused
    else:
        raise ValueError("risk_by must be instability or fused")

    order = np.argsort(ss["_risk"].to_numpy(), kind="mergesort")
    ss_ord = ss.iloc[order].reset_index(drop=True)
    n = len(ss_ord)
    out_rows: List[Dict[str, object]] = []
    for step in range(1, n_grid + 1):
        frac = step / n_grid
        k = max(1, int(np.ceil(frac * n)))
        kept = ss_ord.iloc[:k]
        kept_paths = set(kept["image_path"].astype(str))
        g_sub = g[g["image_path"].astype(str).isin(kept_paths)]
        p_sub = _build_pairs_for_setting(g_sub, max_pairs=max_pairs, rng=rng)
        if len(p_sub) == 0:
            mean_risk = float("nan")
            npairs = 0
        else:
            mean_risk = float(p_sub["pairwise_error"].mean())
            npairs = int(len(p_sub))
        n_an = int((g_sub["label"] == 1).sum())
        n_neg = int((g_sub["label"] == 0).sum())
        out_rows.append(
            {
                "coverage": float(k / n),
                "mean_pairwise_risk": mean_risk,
                "n_pairs": npairs,
                "n_images_kept": k,
                "n_anomaly_kept": n_an,
                "n_normal_kept": n_neg,
                "risk_score": risk_by,
            }
        )
    out = pd.DataFrame(out_rows)
    return out.sort_values("coverage").drop_duplicates(subset=["coverage"], keep="last").reset_index(drop=True)


def _pooled_risk_for_plot(rc_parts: List[pd.DataFrame], n_grid: int) -> pd.DataFrame:
    """Interpolate each setting's (coverage, risk) curve to a common grid; nan-mean across settings."""
    targets = np.linspace(1.0 / n_grid, 1.0, n_grid)
    rows: List[Dict[str, float]] = []
    for t in targets:
        vals: List[float] = []
        for rc in rc_parts:
            c = rc["coverage"].to_numpy(dtype=float)
            r = rc["mean_pairwise_risk"].to_numpy(dtype=float)
            if len(c) == 0:
                continue
            order = np.argsort(c)
            c = c[order]
            r = r[order]
            if len(c) == 1:
                vals.append(float(r[0]))
            else:
                vals.append(float(np.interp(t, c, r, left=float("nan"), right=float("nan"))))
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        pooled = float(finite.mean()) if finite.size > 0 else float("nan")
        rows.append({"target_coverage": float(t), "mean_pairwise_risk": pooled})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("outputs/cached_results/raw_scores/promptad"))
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/cached_results/sec3_promptad"))
    ap.add_argument("--fig-dir", type=Path, default=Path("outputs/figures/sec3_promptad"))
    ap.add_argument("--near-auroc", type=float, default=0.01, help="Max |ΔAUROC| for same-AUROC candidates")
    ap.add_argument("--min-inst-gap", type=float, default=0.03, help="Min |Δmean pairwise I| for flagged seed pairs")
    ap.add_argument(
        "--max-pairs-per-setting",
        type=int,
        default=None,
        help="Cap pairs per (dataset,category,shot,seed); uniform subsample if exceeded",
    )
    ap.add_argument("--pair-sampling-seed", type=int, default=42, help="RNG seed for pair subsampling")
    ap.add_argument(
        "--risk-by",
        choices=("instability", "fused"),
        default="instability",
        help="Sample ordering for risk–coverage accepted subset",
    )
    ap.add_argument("--risk-coverage-grid", type=int, default=20, help="Number of coverage steps (pooled over settings)")
    args = ap.parse_args()

    raw_dir = args.raw_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    fig_dir = args.fig_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.pair_sampling_seed))

    df = _load_wide_or_long(raw_dir)
    need = {"sample_id", "label", "fused_score", "semantic_score", "visual_score", "dataset", "category", "shot", "seed"}
    if not need.issubset(df.columns):
        raise SystemExit(f"raw frame missing columns: {sorted(need - set(df.columns))}")

    setting_keys = ["dataset", "category", "shot", "seed"]
    all_pair_parts: List[pd.DataFrame] = []
    setting_rows: List[Dict[str, object]] = []
    sample_parts: List[pd.DataFrame] = []
    total_pairs_raw = 0
    total_pairs_kept = 0
    any_subsample = False

    for key, g in df.groupby(setting_keys, sort=False):
        d, cat, shot, seed = key
        n_pos = int((g["label"] == 1).sum())
        n_neg = int((g["label"] == 0).sum())
        raw_n_pairs = n_pos * n_neg
        total_pairs_raw += raw_n_pairs
        p_df = _build_pairs_for_setting(g, max_pairs=args.max_pairs_per_setting, rng=rng)
        if raw_n_pairs > 0 and args.max_pairs_per_setting is not None and raw_n_pairs > args.max_pairs_per_setting:
            any_subsample = True
        total_pairs_kept += len(p_df)
        if len(p_df) == 0:
            setting_rows.append(
                {
                    "dataset": d,
                    "category": cat,
                    "shot": int(shot),
                    "seed": int(seed),
                    "n_images": int(len(g)),
                    "n_anomaly": n_pos,
                    "n_normal": n_neg,
                    "n_pairs": 0,
                    "setting_auroc": float("nan"),
                    "mean_pairwise_instability": float("nan"),
                    "mean_pairwise_error": float("nan"),
                }
            )
            continue

        setting_auroc = float(p_df["z_fused"].mean())
        mean_i = float(p_df["pairwise_instability"].mean())
        mean_e = float(p_df["pairwise_error"].mean())
        setting_rows.append(
            {
                "dataset": d,
                "category": cat,
                "shot": int(shot),
                "seed": int(seed),
                "n_images": int(len(g)),
                "n_anomaly": n_pos,
                "n_normal": n_neg,
                "n_pairs": int(len(p_df)),
                "setting_auroc": setting_auroc,
                "mean_pairwise_instability": mean_i,
                "mean_pairwise_error": mean_e,
            }
        )
        all_pair_parts.append(p_df)
        sm = _sample_instability_from_pairs(p_df, g)
        sample_parts.append(sm)

    pairwise_all = pd.concat(all_pair_parts, ignore_index=True) if all_pair_parts else pd.DataFrame()
    pairwise_all.to_csv(cache_dir / "pairwise_metrics.csv", index=False)

    met = pd.DataFrame(setting_rows)
    met.to_csv(cache_dir / "setting_level_metrics.csv", index=False)

    sample_all = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    sample_all.to_csv(cache_dir / "sample_level_metrics.csv", index=False)

    # same-AUROC candidate pairs across seeds within (dataset, category, shot)
    pair_rows: List[Dict[str, object]] = []
    for (d, cat, shot), g in met.groupby(["dataset", "category", "shot"], sort=False):
        seeds = g.sort_values("seed")
        aucs = seeds["setting_auroc"].to_numpy(dtype=float)
        insts = seeds["mean_pairwise_instability"].to_numpy(dtype=float)
        sd = seeds["seed"].to_numpy(dtype=int)
        n = len(seeds)
        for i in range(n):
            for j in range(i + 1, n):
                if not (np.isfinite(aucs[i]) and np.isfinite(aucs[j])):
                    continue
                if abs(aucs[i] - aucs[j]) <= args.near_auroc and abs(insts[i] - insts[j]) >= args.min_inst_gap:
                    pair_rows.append(
                        {
                            "dataset": d,
                            "category": cat,
                            "shot": int(shot),
                            "run_a": f"{d}__{cat}__k{shot}_s{sd[i]}",
                            "run_b": f"{d}__{cat}__k{shot}_s{sd[j]}",
                            "auroc_gap": float(abs(aucs[i] - aucs[j])),
                            "instability_gap": float(abs(insts[i] - insts[j])),
                            "seed_a": int(sd[i]),
                            "seed_b": int(sd[j]),
                        }
                    )
    same_au = pd.DataFrame(pair_rows)
    same_au.to_csv(cache_dir / "same_auroc_instability_pairs.csv", index=False)

    # Risk–coverage: per setting, accepted subset by sample ordering; pairs only inside subset (pairwise risk).
    rc_mean = pd.DataFrame()
    rc_all = pd.DataFrame()
    if len(sample_all) > 0:
        rc_parts: List[pd.DataFrame] = []
        for key, g in df.groupby(setting_keys, sort=False):
            d, cat, shot, seed = key
            if len(g) < 2 or (g["label"] == 1).sum() == 0 or (g["label"] == 0).sum() == 0:
                continue
            join_cols = ["dataset", "category", "shot", "seed", "sample_id", "image_path"]
            extra = [
                c
                for c in (
                    "mean_pairwise_instability",
                    "mean_pairwise_error",
                    "n_pairs",
                    "role_in_pairs",
                )
                if c in sample_all.columns
            ]
            sub_s = g.merge(sample_all[join_cols + extra], on=join_cols, how="left")
            rc = _risk_coverage_curve(
                g,
                sub_s,
                risk_by=args.risk_by,
                max_pairs=args.max_pairs_per_setting,
                rng=rng,
                n_grid=args.risk_coverage_grid,
            )
            rc["dataset"] = d
            rc["category"] = cat
            rc["shot"] = int(shot)
            rc["seed"] = int(seed)
            rc_parts.append(rc)
        if rc_parts:
            rc_all = pd.concat(rc_parts, ignore_index=True)
            rc_all.to_csv(cache_dir / "risk_coverage.csv", index=False)
            rc_mean = _pooled_risk_for_plot(rc_parts, args.risk_coverage_grid)
        else:
            pd.DataFrame().to_csv(cache_dir / "risk_coverage.csv", index=False)
    else:
        pd.DataFrame().to_csv(cache_dir / "risk_coverage.csv", index=False)

    # --- Figures (pairwise semantics) ---
    plt.rcParams.update({"font.size": 10})

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sc = ax.scatter(met["setting_auroc"], met["mean_pairwise_instability"], c=met["seed"], cmap="viridis", alpha=0.75, s=42)
    plt.colorbar(sc, ax=ax, label="seed")
    ax.set_xlabel("setting AUROC = mean(z_fused) over pairs")
    ax.set_ylabel("mean pairwise_instability = Var([z_sem, z_vis])")
    ax.set_title("Sec 3.1.1 PromptAD (pairwise): AUROC vs instability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "scatter_auroc_vs_instability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if len(pairwise_all) > 0:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.hist(pairwise_all["pairwise_instability"].to_numpy(dtype=float), bins=40, color="steelblue", edgecolor="white", alpha=0.9)
        ax.set_xlabel("pairwise_instability (per anomaly–normal pair)")
        ax.set_ylabel("count")
        ax.set_title("Pairwise instability distribution (pooled)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "hist_instability_distribution.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if len(sample_all) > 0:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.scatter(
            sample_all["mean_pairwise_instability"],
            sample_all["mean_pairwise_error"],
            alpha=0.35,
            s=18,
            c="darkred",
        )
        ax.set_xlabel("sample mean pairwise_instability")
        ax.set_ylabel("sample mean pairwise_error")
        ax.set_title("Instability vs error (sample-level aggregates)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "scatter_instability_vs_ranking_error.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if len(rc_mean) > 0:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.plot(rc_mean["target_coverage"], rc_mean["mean_pairwise_risk"], color="darkgreen", lw=2)
        ax.set_xlabel("target coverage (interpolated across settings)")
        ax.set_ylabel("mean pairwise_error on accepted subset")
        ax.set_title(f"Risk–coverage (pairwise; accept by {args.risk_by})")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "risk_coverage_ranking_error.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if len(same_au) > 0:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.scatter(same_au["auroc_gap"], same_au["instability_gap"], alpha=0.8, s=48, c="purple")
        ax.set_xlabel("|Δ setting AUROC|")
        ax.set_ylabel("|Δ mean pairwise_instability|")
        ax.set_title("Near-equal AUROC seed pairs (pairwise definition)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "scatter_same_auroc_instability_gaps.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "definition": "pairwise_anomaly_normal_z_sem_z_vis_var_instability",
        "setting_auroc_note": "mean(z_fused) over anomaly-normal pairs (one-sided dominance; ties -> z=0)",
        "n_settings": int(len(met)),
        "total_pairs_enumerated_raw": int(total_pairs_raw),
        "total_pairs_in_tables": int(total_pairs_kept),
        "max_pairs_per_setting": args.max_pairs_per_setting,
        "pair_sampling_seed": int(args.pair_sampling_seed),
        "pair_subsampling_used": bool(any_subsample),
        "near_auroc_threshold": args.near_auroc,
        "min_inst_gap": args.min_inst_gap,
        "risk_coverage_risk_by": args.risk_by,
        "risk_coverage_csv": "long_rows_per_setting_and_coverage_step_figure_uses_interpolated_pooled_curve",
        "raw_dir": str(raw_dir),
    }
    (cache_dir / "sec3_promptad_from_raw_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (fig_dir / "from_raw_done.txt").write_text(
        "Sec 3.1.1 pairwise analysis from unified raw (no PromptAD train/infer).\n",
        encoding="utf-8",
    )
    print(f"wrote {cache_dir}/pairwise_metrics.csv ({len(pairwise_all)} rows)")
    print(f"wrote {cache_dir}/setting_level_metrics.csv ({len(met)} rows)")
    print(f"wrote {cache_dir}/sample_level_metrics.csv ({len(sample_all)} rows)")
    print(f"wrote {cache_dir}/same_auroc_instability_pairs.csv ({len(same_au)} rows)")
    print(f"wrote {cache_dir}/risk_coverage.csv (long format, {len(rc_all)} rows)")
    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
