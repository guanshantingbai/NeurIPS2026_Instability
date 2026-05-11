#!/usr/bin/env python3
"""
PromptAD instability strengthening experiments (Exp 1–6 + README + warnings).

Reads pilot outputs + per-image CSVs; writes to
PromptAD/result_analysis/promptad_strengthening/
"""
from __future__ import annotations

import argparse
import os
import warnings as py_warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None  # type: ignore

from pilot_instability_aware_selection import (  # noqa: E402
    EPS,
    FINAL_SCORE_CANDIDATES,
    LABEL_CANDIDATES,
    PATH_CANDIDATES,
    SEM_SCORE_CANDIDATES,
    VIS_SCORE_CANDIDATES,
    _discover_per_sample_csvs,
    _first_existing_column,
    _parse_setting_seed_from_path,
    _to_label01,
)

COVERAGES = (0.9, 0.8, 0.7, 0.5, 0.3)
EPSILONS = (0.002, 0.005)
BOOTSTRAP_N = 1000
RNG_SEED = 42


def _savefig_both(fig: plt.Figure, path_no_ext: str) -> None:
    fig.savefig(path_no_ext + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


class WarnLog:
    def __init__(self) -> None:
        self.rows: List[Dict[str, str]] = []

    def add(self, experiment: str, message: str, detail: str = "") -> None:
        self.rows.append({"experiment": experiment, "message": message, "detail": detail})


def _load_standardized_per_sample(df: pd.DataFrame) -> pd.DataFrame:
    label_col = _first_existing_column(df, LABEL_CANDIDATES)
    sem_col = _first_existing_column(df, SEM_SCORE_CANDIDATES)
    vis_col = _first_existing_column(df, VIS_SCORE_CANDIDATES)
    final_col = _first_existing_column(df, FINAL_SCORE_CANDIDATES)
    img_col = _first_existing_column(df, PATH_CANDIDATES)
    if label_col is None or sem_col is None or vis_col is None:
        raise ValueError("Missing label/semantic/visual")
    work = pd.DataFrame()
    work["image_path"] = df[img_col].astype(str) if img_col is not None else np.arange(len(df)).astype(str)
    work["image_label"] = _to_label01(df[label_col])
    work["semantic_score"] = pd.to_numeric(df[sem_col], errors="coerce")
    work["visual_score"] = pd.to_numeric(df[vis_col], errors="coerce")
    if final_col is not None:
        work["final_score"] = pd.to_numeric(df[final_col], errors="coerce")
    else:
        work["final_score"] = np.nan
    miss = work["final_score"].isna()
    if miss.any():
        a = work.loc[miss, "semantic_score"].to_numpy(dtype=np.float64)
        b = work.loc[miss, "visual_score"].to_numpy(dtype=np.float64)
        work.loc[miss, "final_score"] = (2.0 * a * b) / (a + b + EPS)
    work = work.dropna(subset=["image_label", "semantic_score", "visual_score", "final_score"]).copy()
    work["image_label"] = work["image_label"].astype(int)
    work = work[(work["image_label"] == 0) | (work["image_label"] == 1)]
    return work


@dataclass
class PairwisePack:
    margin: np.ndarray
    abs_margin: np.ndarray
    error: np.ndarray
    i_flip: np.ndarray


def _pairwise_pack(sem: np.ndarray, vis: np.ndarray, final: np.ndarray, y: np.ndarray) -> Optional[PairwisePack]:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    p, n = int(pos_idx.size), int(neg_idx.size)
    if p == 0 or n == 0:
        return None
    fp, fn = final[pos_idx], final[neg_idx]
    sp, sn = sem[pos_idx], sem[neg_idx]
    vp, vn = vis[pos_idx], vis[neg_idx]
    margin = (fp[:, None] - fn[None, :]).ravel()
    z_sem = sp[:, None] > sn[None, :]
    z_vis = vp[:, None] > vn[None, :]
    i_flip = (z_sem != z_vis).astype(np.float64).ravel()
    err = (fp[:, None] <= fn[None, :]).astype(np.float64).ravel()
    return PairwisePack(margin=margin, abs_margin=np.abs(margin), error=err, i_flip=i_flip)


def _tertile_labels(abs_m: np.ndarray) -> np.ndarray:
    if abs_m.size < 3:
        return np.full(abs_m.size, -1, dtype=np.int8)
    q1, q2 = np.quantile(abs_m, [1 / 3, 2 / 3])
    lab = np.zeros(abs_m.size, dtype=np.int8)
    lab[abs_m > q1] = 1
    lab[abs_m > q2] = 2
    return lab


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    if scipy_stats is None:
        return float("nan"), int(len(x))
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n < 3:
        return float("nan"), n
    with py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore")
        r, _ = scipy_stats.spearmanr(x, y)
    return float(r) if np.isfinite(r) else float("nan"), n


def _error_concentration(sample_error: np.ndarray, signal_high_risky: np.ndarray, alpha: float) -> float:
    te = float(np.sum(sample_error))
    if te <= 0 or sample_error.size == 0:
        return float("nan")
    n = len(sample_error)
    k = max(1, int(np.ceil(alpha * n)))
    order = np.argsort(-signal_high_risky, kind="mergesort")
    top = order[:k]
    return float(np.sum(sample_error[top]) / te)


def _bootstrap_resample_settings(sub: pd.DataFrame, metric_fn: Any, rng: np.random.Generator, n_boot: int) -> Tuple[float, float, float]:
    keys = sub[["dataset", "class", "k"]].drop_duplicates().to_records(index=False).tolist()
    keys = [(str(a), str(b), int(c)) for a, b, c in keys]
    if not keys:
        return float("nan"), float("nan"), float("nan")

    def _subset(picked: List[Tuple[str, str, int]]) -> pd.DataFrame:
        parts = [sub[(sub["dataset"] == k[0]) & (sub["class"] == k[1]) & (sub["k"] == k[2])] for k in picked]
        return pd.concat(parts, axis=0) if parts else sub.iloc[0:0]

    obs = float(metric_fn(sub))
    if len(keys) < 2:
        return obs, float("nan"), float("nan")
    boots: List[float] = []
    for _ in range(n_boot):
        idx = rng.choice(len(keys), size=len(keys), replace=True)
        pick = [keys[i] for i in idx]
        boots.append(float(metric_fn(_subset(pick))))
    boots_a = np.array([b for b in boots if np.isfinite(b)], dtype=np.float64)
    if boots_a.size == 0:
        return obs, float("nan"), float("nan")
    return obs, float(np.quantile(boots_a, 0.025)), float(np.quantile(boots_a, 0.975))


def _write_readme(
    out_dir: str,
    alt_summ_df: pd.DataFrame,
    summ1_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    summ_sel: pd.DataFrame,
    reg_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    claim_block: str,
) -> None:
    if alt_summ_df.empty:
        inst_s = gap_s = "n/a"
    else:
        inst_row = alt_summ_df[alt_summ_df["signal"] == "instability"].iloc[0]
        gap_row = alt_summ_df[alt_summ_df["signal"] == "gap"].iloc[0]
        inst_s = f"{inst_row['mean_spearman']:.4f} (median {inst_row['median_spearman']:.4f})"
        gap_s = f"{gap_row['mean_spearman']:.4f} (median {gap_row['median_spearman']:.4f})"
    lines = [
        "# PromptAD instability strengthening experiments",
        "",
        "## What each experiment does",
        "",
        "1. **Controlled margin (Exp1)** — Within tertiles of |s_final(x+)-s_final(x-)|, compare mean pairwise ranking error for low-I (I_flip=0) vs high-I (I_flip=1) pairs.",
        "2. **Alternative signals (Exp2)** — Per sample: instability, branch gap, score variance, confidence, margin-uncertainty; Spearman vs sample_error and error-concentration at α∈{0.1,0.2,0.3}.",
        "3. **Failure-conditioned separation (Exp3)** — On `failure_analysis.csv` rows: failure = gap_auc>0.01; compare mean signal (AUROC seed) failure vs non-failure.",
        "4. **Selection baselines (Exp4)** — Same candidate set as pilot: near-best AUROC + argmin/argmax baseline signals vs Low-I and Oracle; gap_to_oracle.",
        "5. **Instability regimes (Exp5)** — zero_I / low_I / mid_I / high_I from global distribution of setting instability (AUROC seed); failure rate and gap_reduction by regime.",
        "6. **Bootstrap / tests (Exp6)** — Resample **settings** (not pairs); 95% CI for separation, conditional delta, gap reduction; paired Wilcoxon AUC vs Low-I risk where applicable.",
        "",
        "## Key numbers (see CSVs for full tables)",
        "",
        f"- Spearman (instability vs sample_error) mean: **{inst_s}**; gap signal: **{gap_s}**.",
        "",
        "## Automated claim notes (this run)",
        "",
        claim_block,
        "",
        "## Claim checklist (see CSVs for details)",
        "",
        "| Claim | Supported? | Notes |",
        "|-------|------------|--------|",
        "| **1** Same margin regime: high-I higher error than low-I | See `controlled_margin_summary.csv` | error_gap_high_minus_low |",
        "| **2** Instability vs alternative signals | `alternative_signal_summary.csv` | Spearman + concentration |",
        "| **3** Failure-conditioned separation | `failure_conditioned_signal_analysis.csv` | direction_correct |",
        "| **4** Regimes vs failure | `instability_regime_analysis.csv` | high_I vs zero_I |",
        "| **5** Bootstrap / Wilcoxon | `statistical_stability_summary.csv` | CI excludes 0? |",
        "",
        "## Boundary: coverage=0.3",
        "",
        "Pilot and failure-driven analyses already show **coverage=0.3** can reverse gains; treat as **boundary regime** for selective rejection.",
        "",
        "## Outputs",
        "",
        "CSVs: controlled_margin_*, alternative_signal_*, failure_conditioned_*, selection_baseline_*, instability_regime_*, statistical_stability_*, warnings.csv",
        "Figures: each saved as **.png** and **.pdf**.",
    ]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    pilot_dir = os.path.abspath(args.pilot_dir)
    failure_dir = os.path.join(pilot_dir, "failure_driven")
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    wlog = WarnLog()
    rng = np.random.default_rng(RNG_SEED)

    per_seed = pd.read_csv(os.path.join(pilot_dir, "per_seed_metrics.csv"))
    if "setting_instability_flip" in per_seed.columns and "setting_instability" not in per_seed.columns:
        per_seed = per_seed.rename(columns={"setting_instability_flip": "setting_instability"})

    failure_analysis = pd.read_csv(os.path.join(failure_dir, "failure_analysis.csv"))

    # Backward compatibility: some failure_analysis.csv versions do not contain seed_auc.
    # Recover seed_auc from selection_summary files first; fallback to AUROC matching.
    if "seed_auc" not in failure_analysis.columns:
        sel_parts: List[pd.DataFrame] = []
        for eps in EPSILONS:
            p = os.path.join(pilot_dir, f"selection_summary_epsilon_{eps:.3f}.csv")
            if os.path.isfile(p):
                try:
                    sel_parts.append(pd.read_csv(p))
                except Exception as e:  # noqa: BLE001
                    wlog.add("compat", "cannot read selection summary", f"{p}: {e}")
        if sel_parts:
            sel_all = pd.concat(sel_parts, axis=0, ignore_index=True)
            need_cols = ["dataset", "class", "k", "coverage", "epsilon", "seed_auc", "auroc_auc_seed"]
            ok_cols = [c for c in need_cols if c in sel_all.columns]
            sel_key = sel_all[ok_cols].drop_duplicates()
            failure_analysis = failure_analysis.merge(
                sel_key,
                on=[c for c in ["dataset", "class", "k", "coverage", "epsilon", "auroc_auc_seed"] if c in sel_key.columns and c in failure_analysis.columns],
                how="left",
            )
        if "seed_auc" not in failure_analysis.columns:
            failure_analysis["seed_auc"] = np.nan
        miss_mask = failure_analysis["seed_auc"].isna()
        if miss_mask.any():
            # Fallback by matching AUROC within per_seed table per setting.
            tmp = per_seed[["dataset", "class", "k", "seed", "auroc"]].copy()
            tmp = tmp.rename(columns={"seed": "seed_auc", "auroc": "auroc_seed_table"})
            # Cartesian merge on setting then pick closest AUROC
            fa_m = failure_analysis[miss_mask].merge(tmp, on=["dataset", "class", "k"], how="left")
            if not fa_m.empty:
                fa_m["dist"] = np.abs(pd.to_numeric(fa_m["auroc_auc_seed"], errors="coerce") - pd.to_numeric(fa_m["auroc_seed_table"], errors="coerce"))
                fa_m = fa_m.sort_values(["dataset", "class", "k", "coverage", "epsilon", "dist", "seed_auc"], kind="mergesort")
                picked = fa_m.groupby(["dataset", "class", "k", "coverage", "epsilon"], as_index=False).first()
                picked = picked[["dataset", "class", "k", "coverage", "epsilon", "seed_auc"]]
                failure_analysis = failure_analysis.merge(
                    picked,
                    on=["dataset", "class", "k", "coverage", "epsilon"],
                    how="left",
                    suffixes=("", "_fill"),
                )
                failure_analysis["seed_auc"] = failure_analysis["seed_auc"].fillna(failure_analysis["seed_auc_fill"])
                failure_analysis = failure_analysis.drop(columns=["seed_auc_fill"], errors="ignore")
        if failure_analysis["seed_auc"].isna().any():
            n_miss = int(failure_analysis["seed_auc"].isna().sum())
            wlog.add("compat", "seed_auc still missing after recovery", f"n_missing={n_miss}")
        failure_analysis["seed_auc"] = pd.to_numeric(failure_analysis["seed_auc"], errors="coerce")

    promptad_root = os.path.abspath(args.promptad_root)
    csv_paths = _discover_per_sample_csvs(promptad_root)
    seen: set[Tuple[str, str, int, int]] = set()
    sample_rows: List[Dict[str, Any]] = []
    exp1_rows: List[Dict[str, Any]] = []
    seed_agg: Dict[Tuple[str, str, int, int], Dict[str, float]] = {}

    for path in csv_paths:
        parsed = _parse_setting_seed_from_path(path)
        if parsed is None:
            continue
        key = (parsed.dataset, parsed.cls, parsed.k, parsed.seed)
        if key in seen:
            continue
        try:
            raw = pd.read_csv(path)
            work = _load_standardized_per_sample(raw)
        except Exception as e:  # noqa: BLE001
            wlog.add("load", str(e), path)
            continue
        sem = work["semantic_score"].to_numpy(dtype=np.float64)
        vis = work["visual_score"].to_numpy(dtype=np.float64)
        final = work["final_score"].to_numpy(dtype=np.float64)
        y = work["image_label"].to_numpy(dtype=np.int8)
        pk = _pairwise_pack(sem, vis, final, y)
        if pk is None:
            wlog.add("pairwise", "missing pos/neg", path)
            continue
        seen.add(key)

        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        sem_pos, sem_neg = sem[pos_idx], sem[neg_idx]
        vis_pos, vis_neg = vis[pos_idx], vis[neg_idx]
        fp, fn = final[pos_idx], final[neg_idx]
        z_sem = sem_pos[:, None] > sem_neg[None, :]
        z_vis = vis_pos[:, None] > vis_neg[None, :]
        flip_m = (z_sem != z_vis).astype(np.float64)
        err_m = (fp[:, None] <= fn[None, :]).astype(np.float64)
        margin_m = (fp[:, None] - fn[None, :]).astype(np.float64)

        inst_s = np.zeros(len(work), dtype=np.float64)
        err_s = np.zeros(len(work), dtype=np.float64)
        mean_abs_margin_s = np.zeros(len(work), dtype=np.float64)
        inst_s[pos_idx] = np.mean(flip_m, axis=1)
        inst_s[neg_idx] = np.mean(flip_m, axis=0)
        err_s[pos_idx] = np.mean(err_m, axis=1)
        err_s[neg_idx] = np.mean(err_m, axis=0)
        mean_abs_margin_s[pos_idx] = np.mean(np.abs(margin_m), axis=1)
        mean_abs_margin_s[neg_idx] = np.mean(np.abs(margin_m), axis=0)

        gap = np.abs(sem - vis)
        m2 = 0.5 * (sem + vis)
        score_var = 0.5 * ((sem - m2) ** 2 + (vis - m2) ** 2)
        med_f = float(np.median(final))
        ranks = pd.Series(final).rank(method="average").to_numpy(dtype=np.float64)
        conf_median = np.abs(final - med_f)
        conf_rank = np.abs(ranks / float(len(final)) - 0.5)
        uncert_conf_median = -conf_median
        uncert_conf_rank = -conf_rank
        uncert_margin = -mean_abs_margin_s

        for i in range(len(work)):
            sample_rows.append(
                {
                    "dataset": parsed.dataset,
                    "class": parsed.cls,
                    "k": int(parsed.k),
                    "seed": int(parsed.seed),
                    "image_path": work["image_path"].iloc[i],
                    "image_label": int(work["image_label"].iloc[i]),
                    "sample_error": float(err_s[i]),
                    "instability": float(inst_s[i]),
                    "gap": float(gap[i]),
                    "score_var": float(score_var[i]),
                    "conf_median": float(conf_median[i]),
                    "conf_rank": float(conf_rank[i]),
                    "mean_abs_margin": float(mean_abs_margin_s[i]),
                    "uncert_conf_median": float(uncert_conf_median[i]),
                    "uncert_conf_rank": float(uncert_conf_rank[i]),
                    "uncert_margin": float(uncert_margin[i]),
                }
            )

        mean_gap = float(np.mean(gap))
        mean_var = float(np.mean(score_var))
        mean_conf_m = float(np.mean(conf_median))
        mean_conf_r = float(np.mean(conf_rank))
        mean_margin = float(np.mean(mean_abs_margin_s))
        seed_agg[key] = {
            "mean_gap": mean_gap,
            "mean_score_var": mean_var,
            "mean_conf_median": mean_conf_m,
            "mean_conf_rank": mean_conf_r,
            "mean_sample_margin": mean_margin,
            "setting_instability": float(np.mean(flip_m)),
        }

        tert = _tertile_labels(pk.abs_margin)
        margin_bucket_map = {0: "low_margin", 1: "mid_margin", 2: "high_margin"}
        for bi in range(3):
            mask_b = tert == bi
            if not np.any(mask_b):
                wlog.add("exp1", f"empty margin bucket {bi}", path)
                continue
            i_sub = pk.i_flip[mask_b]
            e_sub = pk.error[mask_b]
            am_sub = pk.abs_margin[mask_b]
            for igroup, imask in [("low_I", i_sub < 0.5), ("high_I", i_sub > 0.5)]:
                m2b = imask
                if not np.any(m2b):
                    wlog.add("exp1", f"empty {igroup} in {margin_bucket_map[bi]}", path)
                    continue
                exp1_rows.append(
                    {
                        "dataset": parsed.dataset,
                        "class": parsed.cls,
                        "k": int(parsed.k),
                        "seed": int(parsed.seed),
                        "margin_bucket": margin_bucket_map[bi],
                        "instability_group": igroup,
                        "n_pairs": int(np.sum(m2b)),
                        "mean_error": float(np.mean(e_sub[m2b])),
                        "mean_abs_margin": float(np.mean(am_sub[m2b])),
                        "mean_instability": float(np.mean(i_sub[m2b])),
                    }
                )

    pd.DataFrame(sample_rows).to_csv(os.path.join(out_dir, "alternative_signal_sample_metrics.csv"), index=False)
    exp1_df = pd.DataFrame(exp1_rows)
    exp1_df.to_csv(os.path.join(out_dir, "controlled_margin_pairwise.csv"), index=False)

    summ1_rows: List[Dict[str, Any]] = []
    for bucket, g in exp1_df.groupby("margin_bucket", sort=False):
        low = g[g["instability_group"] == "low_I"]
        high = g[g["instability_group"] == "high_I"]
        n_low = int(low["n_pairs"].sum()) if len(low) else 0
        n_high = int(high["n_pairs"].sum()) if len(high) else 0
        err_low = float(np.average(low["mean_error"], weights=low["n_pairs"])) if n_low > 0 else float("nan")
        err_high = float(np.average(high["mean_error"], weights=high["n_pairs"])) if n_high > 0 else float("nan")
        mam_low = float(np.average(low["mean_abs_margin"], weights=low["n_pairs"])) if n_low > 0 else float("nan")
        mam_high = float(np.average(high["mean_abs_margin"], weights=high["n_pairs"])) if n_high > 0 else float("nan")
        summ1_rows.append(
            {
                "margin_bucket": bucket,
                "n_pairs_low_I": n_low,
                "n_pairs_high_I": n_high,
                "error_low_I": err_low,
                "error_high_I": err_high,
                "error_gap_high_minus_low": (err_high - err_low) if np.isfinite(err_high) and np.isfinite(err_low) else float("nan"),
                "mean_abs_margin_low_I": mam_low,
                "mean_abs_margin_high_I": mam_high,
            }
        )
    summ1_df = pd.DataFrame(summ1_rows)
    summ1_df.to_csv(os.path.join(out_dir, "controlled_margin_summary.csv"), index=False)

    order = ["low_margin", "mid_margin", "high_margin"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(order))
    w = 0.35
    err_lo, err_hi = [], []
    for b in order:
        subb = summ1_df[summ1_df["margin_bucket"] == b]
        err_lo.append(float(subb["error_low_I"].iloc[0]) if len(subb) else np.nan)
        err_hi.append(float(subb["error_high_I"].iloc[0]) if len(subb) else np.nan)
    ax.bar(x - w / 2, err_lo, width=w, label="low-I")
    ax.bar(x + w / 2, err_hi, width=w, label="high-I")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("mean ranking error")
    ax.set_xlabel("margin bucket")
    ax.legend()
    ax.set_title("Controlled margin: error by instability group")
    _savefig_both(fig, os.path.join(out_dir, "controlled_margin_error_bar"))

    # Exp2
    sm = pd.DataFrame(sample_rows)
    if sm.empty:
        pd.DataFrame(wlog.rows).to_csv(os.path.join(out_dir, "warnings.csv"), index=False)
        raise SystemExit("No per-sample rows loaded; check per_sample CSV paths and pilot_dir.")

    signals = [
        ("instability", "instability"),
        ("gap", "gap"),
        ("score_var", "score_var"),
        ("uncert_conf_median", "uncert_conf_median"),
        ("uncert_conf_rank", "uncert_conf_rank"),
        ("uncert_margin", "uncert_margin"),
    ]
    corr_rows: List[Dict[str, Any]] = []
    conc_rows: List[Dict[str, Any]] = []
    for (d, c, k, s), grp in sm.groupby(["dataset", "class", "k", "seed"], sort=False):
        se = grp["sample_error"].to_numpy(dtype=np.float64)
        for sig_name, col in signals:
            sigv = grp[col].to_numpy(dtype=np.float64)
            rho, n_s = _spearman(sigv, se)
            corr_rows.append({"dataset": d, "class": c, "k": int(k), "seed": int(s), "signal": sig_name, "spearman": rho, "n_samples": n_s})
            for alpha in (0.1, 0.2, 0.3):
                cap = _error_concentration(se, sigv, alpha)
                conc_rows.append(
                    {
                        "dataset": d,
                        "class": c,
                        "k": int(k),
                        "seed": int(s),
                        "signal": sig_name,
                        "alpha": alpha,
                        "error_captured": cap,
                        "random_baseline": alpha,
                        "amplification": cap / alpha if np.isfinite(cap) and alpha > 0 else float("nan"),
                    }
                )

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(out_dir, "alternative_signal_correlation.csv"), index=False)
    conc_df = pd.DataFrame(conc_rows)
    conc_df.to_csv(os.path.join(out_dir, "alternative_signal_concentration.csv"), index=False)

    alt_summ = []
    for sig_name, _ in signals:
        sub = corr_df[corr_df["signal"] == sig_name]
        subc = conc_df[conc_df["signal"] == sig_name]
        alt_summ.append(
            {
                "signal": sig_name,
                "mean_spearman": float(sub["spearman"].mean()),
                "median_spearman": float(sub["spearman"].median()),
                "mean_error_captured_10": float(subc[subc["alpha"] == 0.1]["error_captured"].mean()),
                "mean_error_captured_20": float(subc[subc["alpha"] == 0.2]["error_captured"].mean()),
                "mean_error_captured_30": float(subc[subc["alpha"] == 0.3]["error_captured"].mean()),
                "median_error_captured_20": float(subc[subc["alpha"] == 0.2]["error_captured"].median()),
            }
        )
    alt_summ_df = pd.DataFrame(alt_summ)
    alt_summ_df.to_csv(os.path.join(out_dir, "alternative_signal_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(signals))
    ax.bar(x - 0.2, alt_summ_df["mean_spearman"], width=0.4, label="mean")
    ax.bar(x + 0.2, alt_summ_df["median_spearman"], width=0.4, label="median")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in signals], rotation=25, ha="right")
    ax.set_ylabel("Spearman rho (vs sample_error)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()
    ax.set_title("Alternative signals: Spearman correlation")
    _savefig_both(fig, os.path.join(out_dir, "alternative_signal_spearman_bar"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for sig_name in [s[0] for s in signals]:
        subc = conc_df[conc_df["signal"] == sig_name].groupby("alpha")["error_captured"].mean()
        ax.plot(subc.index, subc.values, marker="o", label=sig_name)
    ax.plot([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], "k--", label="random")
    ax.set_xlabel("alpha (top fraction)")
    ax.set_ylabel("mean error captured")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Error concentration by signal")
    _savefig_both(fig, os.path.join(out_dir, "alternative_signal_error_concentration"))

    # Exp3
    lk = []
    for k, v in seed_agg.items():
        lk.append(
            {
                "dataset": k[0],
                "class": k[1],
                "k": k[2],
                "seed_auc_join": k[3],
                "gap_setting": v["mean_gap"],
                "score_var_setting": v["mean_score_var"],
                "confidence_median_setting": v["mean_conf_median"],
                "confidence_rank_setting": v["mean_conf_rank"],
                "margin_setting": v["mean_sample_margin"],
            }
        )
    lk_df = pd.DataFrame(lk)
    fa = failure_analysis.merge(
        lk_df,
        left_on=["dataset", "class", "k", "seed_auc"],
        right_on=["dataset", "class", "k", "seed_auc_join"],
        how="left",
    ).drop(columns=["seed_auc_join"], errors="ignore")

    signal_defs = [
        ("instability", "instability_auc_seed", True),
        ("gap", "gap_setting", True),
        ("score_var", "score_var_setting", True),
        ("confidence_median", "confidence_median_setting", False),
        ("confidence_rank", "confidence_rank_setting", False),
        ("margin", "margin_setting", False),
    ]
    fc_rows: List[Dict[str, Any]] = []
    for (eps, cov), g in fa.groupby(["epsilon", "coverage"], sort=True):
        g_f = g[g["failure"] == True]
        g_nf = g[g["failure"] == False]
        for sig_name, col, risk_high in signal_defs:
            vf, vnf = g_f[col], g_nf[col]
            vf = pd.to_numeric(vf, errors="coerce").dropna()
            vnf = pd.to_numeric(vnf, errors="coerce").dropna()
            mf, mnf = float(vf.mean()) if len(vf) else float("nan"), float(vnf.mean()) if len(vnf) else float("nan")
            medf = float(vf.median()) if len(vf) else float("nan")
            mednf = float(vnf.median()) if len(vnf) else float("nan")
            diff = mf - mnf if np.isfinite(mf) and np.isfinite(mnf) else float("nan")
            ratio = mf / mnf if mnf not in (0, float("nan")) and np.isfinite(mnf) and np.isfinite(mf) else float("nan")
            direction_ok = (mf > mnf) if risk_high else (mf < mnf)
            if np.isfinite(mf) and np.isfinite(mnf):
                direction_ok = bool(direction_ok)
            else:
                direction_ok = False
            fc_rows.append(
                {
                    "epsilon": float(eps),
                    "coverage": float(cov),
                    "signal": sig_name,
                    "n_failure": int(len(g_f)),
                    "n_non_failure": int(len(g_nf)),
                    "mean_failure": mf,
                    "mean_non_failure": mnf,
                    "median_failure": medf,
                    "median_non_failure": mednf,
                    "diff": diff,
                    "ratio": ratio,
                    "direction_correct": direction_ok,
                }
            )
    fc_df = pd.DataFrame(fc_rows)
    fc_df.to_csv(os.path.join(out_dir, "failure_conditioned_signal_analysis.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for sig in ["instability", "gap", "score_var"]:
        sub = fc_df[(fc_df["signal"] == sig) & (fc_df["epsilon"] == 0.002)]
        ax.plot(sub["coverage"], sub["diff"], marker="o", label=sig)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("coverage")
    ax.set_ylabel("mean_failure - mean_non_failure")
    ax.legend()
    ax.set_title("Failure-conditioned separation (eps=0.002)")
    _savefig_both(fig, os.path.join(out_dir, "failure_conditioned_signal_separation"))

    fig, ax = plt.subplots(figsize=(7, 5))
    fail_c = fa["failure"].map({True: "tab:red", False: "tab:blue"})
    ax.scatter(fa["instability_auc_seed"], fa["gap_auc"], c=fail_c, alpha=0.6, s=22)
    ax.set_xlabel("instability_auc_seed")
    ax.set_ylabel("gap_auc (risk_auc - risk_oracle)")
    ax.set_title("Instability vs failure gap (all eps/cov pooled in file — see CSV for split)")
    _savefig_both(fig, os.path.join(out_dir, "failure_instability_scatter"))

    # Exp4 selection
    cov_cols = {c: f"risk_cov_{c:.1f}" for c in COVERAGES}
    sel_rows: List[Dict[str, Any]] = []
    for (d, c, k), g in per_seed.groupby(["dataset", "class", "k"], sort=False):
        g = g.copy()
        if len(g) < 2:
            continue
        best_auroc = float(g["auroc"].max())
        for eps in EPSILONS:
            cand = g[g["auroc"] >= best_auroc - eps].copy()
            if cand.empty:
                continue

            def _agg_row(seed: int) -> Optional[Dict[str, float]]:
                ag = seed_agg.get((d, c, int(k), int(seed)))
                if ag is None:
                    return None
                r = cand[cand["seed"] == seed].iloc[0]
                out = dict(ag)
                out["auroc"] = float(r["auroc"])
                out["instability"] = float(r["setting_instability"])
                for cov in COVERAGES:
                    col = cov_cols[cov]
                    out[f"risk_{cov}"] = float(r[col]) if col in r.index and pd.notna(r[col]) else float("nan")
                return out

            seeds_c = [int(s) for s in cand["seed"].tolist()]
            rows_c = {s: _agg_row(s) for s in seeds_c}
            rows_c = {s: v for s, v in rows_c.items() if v is not None}
            if not rows_c:
                wlog.add("exp4", "no seed_agg", f"{d},{c},{k}")
                continue

            def _pick(rule: str) -> Tuple[int, Dict[str, float]]:
                items = list(rows_c.items())
                if rule == "AUC":
                    return max(items, key=lambda x: (x[1]["auroc"], -x[1]["instability"], -x[0]))
                if rule == "Low-I":
                    return min(items, key=lambda x: (x[1]["instability"], -x[1]["auroc"], x[0]))
                if rule == "Low-gap":
                    return min(items, key=lambda x: (x[1]["mean_gap"], -x[1]["auroc"], x[0]))
                if rule == "Low-var":
                    return min(items, key=lambda x: (x[1]["mean_score_var"], -x[1]["auroc"], x[0]))
                if rule == "High-conf":
                    return max(items, key=lambda x: (x[1]["mean_conf_median"], x[1]["auroc"], -x[0]))
                if rule == "High-margin":
                    return max(items, key=lambda x: (x[1]["mean_sample_margin"], x[1]["auroc"], -x[0]))
                raise ValueError(rule)

            for cov in COVERAGES:
                risks = {s: rows_c[s][f"risk_{cov}"] for s in rows_c}
                finite = {s: v for s, v in risks.items() if np.isfinite(v)}
                if not finite:
                    continue
                s_oracle = min(finite, key=lambda s: (finite[s], s))
                r_oracle = float(finite[s_oracle])
                for rule in ["AUC", "Low-I", "Low-gap", "Low-var", "High-conf", "High-margin"]:
                    sid, info = _pick(rule)
                    risk = float(info[f"risk_{cov}"])
                    sel_rows.append(
                        {
                            "dataset": d,
                            "class": c,
                            "k": int(k),
                            "epsilon": eps,
                            "coverage": cov,
                            "rule": rule,
                            "selected_seed": sid,
                            "auroc": info["auroc"],
                            "instability": info["instability"],
                            "gap": info["mean_gap"],
                            "score_var": info["mean_score_var"],
                            "confidence": info["mean_conf_median"],
                            "margin": info["mean_sample_margin"],
                            "risk": risk,
                            "oracle_risk": r_oracle,
                            "gap_to_oracle": risk - r_oracle,
                        }
                    )
                info_o = rows_c[s_oracle]
                sel_rows.append(
                    {
                        "dataset": d,
                        "class": c,
                        "k": int(k),
                        "epsilon": eps,
                        "coverage": cov,
                        "rule": "Oracle",
                        "selected_seed": s_oracle,
                        "auroc": info_o["auroc"],
                        "instability": info_o["instability"],
                        "gap": info_o["mean_gap"],
                        "score_var": info_o["mean_score_var"],
                        "confidence": info_o["mean_conf_median"],
                        "margin": info_o["mean_sample_margin"],
                        "risk": r_oracle,
                        "oracle_risk": r_oracle,
                        "gap_to_oracle": 0.0,
                    }
                )

    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(os.path.join(out_dir, "selection_baseline_comparison.csv"), index=False)

    def _summ_sel(sub: pd.DataFrame) -> pd.DataFrame:
        rows_out: List[Dict[str, Any]] = []
        for (eps, cov, rule), gg in sub.groupby(["epsilon", "coverage", "rule"], sort=True):
            gr = gg["gap_to_oracle"]
            auc_part = sub[(sub["rule"] == "AUC") & (sub["epsilon"] == eps) & (sub["coverage"] == cov)][
                ["dataset", "class", "k", "risk"]
            ].rename(columns={"risk": "risk_auc_only"})
            merged = gg.merge(auc_part, on=["dataset", "class", "k"], how="left")
            win = float(np.mean(merged["risk"] < merged["risk_auc_only"])) if len(merged) else float("nan")
            mean_delta = float(np.mean(merged["risk_auc_only"] - merged["risk"])) if len(merged) else float("nan")
            rows_out.append(
                {
                    "epsilon": float(eps),
                    "coverage": float(cov),
                    "rule": rule,
                    "n_settings": int(len(gg)),
                    "mean_risk": float(gg["risk"].mean()),
                    "median_risk": float(gg["risk"].median()),
                    "mean_gap_to_oracle": float(gr.mean()),
                    "median_gap_to_oracle": float(gr.median()),
                    "win_rate_vs_auc": win,
                    "mean_delta_vs_auc": mean_delta,
                }
            )
        return pd.DataFrame(rows_out)

    summ_sel = _summ_sel(sel_df)
    summ_sel.to_csv(os.path.join(out_dir, "selection_baseline_summary.csv"), index=False)

    fa_key = failure_analysis[["dataset", "class", "k", "epsilon", "coverage", "failure"]].copy()
    fa_key["fail"] = fa_key["failure"]
    sel_m = sel_df.merge(fa_key, on=["dataset", "class", "k", "epsilon", "coverage"], how="left")
    sel_fail = sel_m[sel_m["fail"] == True]
    _summ_sel(sel_fail).to_csv(os.path.join(out_dir, "selection_baseline_summary_failure_only.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for rule in ["AUC", "Low-I", "Low-gap", "Low-var", "High-conf", "High-margin", "Oracle"]:
        sub = summ_sel[(summ_sel["rule"] == rule) & (summ_sel["epsilon"] == 0.002)]
        ax.plot(sub["coverage"], sub["mean_risk"], marker="o", label=rule)
    ax.set_xlabel("coverage")
    ax.set_ylabel("mean risk")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Selection rules vs risk (eps=0.002)")
    _savefig_both(fig, os.path.join(out_dir, "selection_baseline_risk_bar"))

    fig, ax = plt.subplots(figsize=(10, 5))
    for rule in ["AUC", "Low-I", "Low-gap", "Oracle"]:
        sub = summ_sel[(summ_sel["rule"] == rule) & (summ_sel["epsilon"] == 0.002)]
        ax.plot(sub["coverage"], sub["mean_gap_to_oracle"], marker="o", label=rule)
    ax.set_xlabel("coverage")
    ax.set_ylabel("mean gap to oracle")
    ax.legend()
    _savefig_both(fig, os.path.join(out_dir, "selection_baseline_oracle_gap_bar"))

    sf = pd.read_csv(os.path.join(out_dir, "selection_baseline_summary_failure_only.csv"))
    fig, ax = plt.subplots(figsize=(10, 5))
    for rule in ["AUC", "Low-I", "Low-gap", "Oracle"]:
        sub = sf[(sf["rule"] == rule) & (sf["epsilon"] == 0.002)]
        if len(sub):
            ax.plot(sub["coverage"], sub["mean_risk"], marker="o", label=rule)
    ax.set_title("Failure-only mean risk (eps=0.002)")
    ax.legend()
    _savefig_both(fig, os.path.join(out_dir, "selection_baseline_failure_only"))

    # Exp5 regimes
    inst_col = "setting_instability" if "setting_instability" in per_seed.columns else "setting_instability_flip"
    inst_all = per_seed[inst_col].to_numpy(dtype=np.float64)
    nonzero = inst_all[inst_all > 1e-8]
    q30, q70 = (np.quantile(nonzero, [0.3, 0.7]) if len(nonzero) else (0.0, 0.0))

    def _regime(i: float) -> str:
        if i <= 1e-8:
            return "zero_I"
        if i <= q30:
            return "low_I"
        if i < q70:
            return "mid_I"
        return "high_I"

    fa2 = failure_analysis.copy()
    fa2["regime"] = fa2["instability_auc_seed"].apply(_regime)
    reg_rows: List[Dict[str, Any]] = []
    for (eps, cov, reg), gg in fa2.groupby(["epsilon", "coverage", "regime"], sort=True):
        reg_rows.append(
            {
                "epsilon": float(eps),
                "coverage": float(cov),
                "regime": reg,
                "n": int(len(gg)),
                "failure_rate": float(gg["failure"].mean()),
                "mean_delta_risk": float(gg["delta_risk"].mean()),
                "median_delta_risk": float(gg["delta_risk"].median()),
                "mean_gap_auc": float(gg["gap_auc"].mean()),
                "mean_gap_stable": float(gg["gap_stable"].mean()),
                "mean_gap_reduction": float(gg["gap_reduction"].mean()),
            }
        )
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(os.path.join(out_dir, "instability_regime_analysis.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for reg in ["zero_I", "low_I", "mid_I", "high_I"]:
        sub = reg_df[(reg_df["regime"] == reg) & (reg_df["epsilon"] == 0.002)]
        ax.plot(sub["coverage"], sub["failure_rate"], marker="o", label=reg)
    ax.set_xlabel("coverage")
    ax.set_ylabel("failure rate")
    ax.legend()
    ax.set_title("Failure rate by instability regime")
    _savefig_both(fig, os.path.join(out_dir, "instability_regime_failure_rate"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for reg in ["zero_I", "low_I", "mid_I", "high_I"]:
        sub = reg_df[(reg_df["regime"] == reg) & (reg_df["epsilon"] == 0.002)]
        ax.plot(sub["coverage"], sub["mean_gap_reduction"], marker="o", label=reg)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()
    ax.set_title("Mean gap reduction by regime")
    _savefig_both(fig, os.path.join(out_dir, "instability_regime_gap_reduction"))

    # Exp6 bootstrap + Wilcoxon
    stat_rows: List[Dict[str, Any]] = []

    def _sep_inst(df: pd.DataFrame) -> float:
        vf = df.loc[df["failure"], "instability_auc_seed"].to_numpy(dtype=np.float64)
        vnf = df.loc[~df["failure"], "instability_auc_seed"].to_numpy(dtype=np.float64)
        if vf.size == 0 or vnf.size == 0:
            return float("nan")
        return float(np.mean(vf) - np.mean(vnf))

    for eps in EPSILONS:
        for cov in COVERAGES:
            sub = fa2[(fa2["epsilon"] == eps) & (fa2["coverage"] == cov)]
            if len(sub) == 0:
                continue
            n_fail = int(sub["failure"].sum())
            warn = "n_failure<10" if n_fail < 10 else ""

            obs_sep, lo_sep, hi_sep = _bootstrap_resample_settings(sub, _sep_inst, rng, BOOTSTRAP_N)
            stat_rows.append(
                {
                    "epsilon": eps,
                    "coverage": cov,
                    "metric": "instability_separation",
                    "n": int(len(sub)),
                    "mean": obs_sep,
                    "ci_low": lo_sep,
                    "ci_high": hi_sep,
                    "p_value_if_available": "",
                    "warning": warn,
                }
            )

            def _mean_dr(df: pd.DataFrame) -> float:
                dr = df.loc[df["failure"], "delta_risk"].to_numpy(dtype=np.float64)
                return float(np.mean(dr)) if dr.size else float("nan")

            def _mean_gr(df: pd.DataFrame) -> float:
                gr = df.loc[df["failure"], "gap_reduction"].to_numpy(dtype=np.float64)
                return float(np.mean(gr)) if gr.size else float("nan")

            obs_d, lo_d, hi_d = _bootstrap_resample_settings(sub, _mean_dr, rng, BOOTSTRAP_N)
            stat_rows.append(
                {
                    "epsilon": eps,
                    "coverage": cov,
                    "metric": "mean_delta_risk_failure",
                    "n": n_fail,
                    "mean": obs_d,
                    "ci_low": lo_d,
                    "ci_high": hi_d,
                    "p_value_if_available": "",
                    "warning": warn,
                }
            )
            obs_g, lo_g, hi_g = _bootstrap_resample_settings(sub, _mean_gr, rng, BOOTSTRAP_N)
            stat_rows.append(
                {
                    "epsilon": eps,
                    "coverage": cov,
                    "metric": "mean_gap_reduction_failure",
                    "n": n_fail,
                    "mean": obs_g,
                    "ci_low": lo_g,
                    "ci_high": hi_g,
                    "p_value_if_available": "",
                    "warning": warn,
                }
            )

            # Baseline: mean gap_to_oracle AUC - mean gap_to_oracle Low-I on failure settings only
            def _gap_diff_for_failure_settings(df: pd.DataFrame) -> float:
                fail_only = df[df["failure"]]
                if fail_only.empty:
                    return float("nan")
                keys_f = fail_only[["dataset", "class", "k"]].drop_duplicates()
                diffs: List[float] = []
                for _, row in keys_f.iterrows():
                    d, c, kk = row["dataset"], row["class"], int(row["k"])
                    a = sel_df[
                        (sel_df["dataset"] == d)
                        & (sel_df["class"] == c)
                        & (sel_df["k"] == kk)
                        & (sel_df["epsilon"] == eps)
                        & (sel_df["coverage"] == cov)
                        & (sel_df["rule"] == "AUC")
                    ]
                    b = sel_df[
                        (sel_df["dataset"] == d)
                        & (sel_df["class"] == c)
                        & (sel_df["k"] == kk)
                        & (sel_df["epsilon"] == eps)
                        & (sel_df["coverage"] == cov)
                        & (sel_df["rule"] == "Low-I")
                    ]
                    if len(a) and len(b):
                        diffs.append(float(a["gap_to_oracle"].iloc[0] - b["gap_to_oracle"].iloc[0]))
                return float(np.mean(diffs)) if diffs else float("nan")

            obs_b, lo_b, hi_b = _bootstrap_resample_settings(sub, _gap_diff_for_failure_settings, rng, BOOTSTRAP_N)
            n_fail_settings = int(sub[sub["failure"]][["dataset", "class", "k"]].drop_duplicates().shape[0])
            stat_rows.append(
                {
                    "epsilon": eps,
                    "coverage": cov,
                    "metric": "mean_gap_to_oracle_auc_minus_lowI_failure_settings",
                    "n": n_fail_settings,
                    "mean": obs_b,
                    "ci_low": lo_b,
                    "ci_high": hi_b,
                    "p_value_if_available": "",
                    "warning": warn,
                }
            )

            # Wilcoxon paired AUC vs Low-I (all settings with both finite)
            a = sel_df[(sel_df["rule"] == "AUC") & (sel_df["epsilon"] == eps) & (sel_df["coverage"] == cov)][["dataset", "class", "k", "risk"]].rename(columns={"risk": "risk_auc"})
            b = sel_df[(sel_df["rule"] == "Low-I") & (sel_df["epsilon"] == eps) & (sel_df["coverage"] == cov)][["dataset", "class", "k", "risk"]].rename(columns={"risk": "risk_lowi"})
            m = a.merge(b, on=["dataset", "class", "k"])
            p_w = ""
            if scipy_stats is not None and len(m) >= 5:
                dff = m["risk_auc"].to_numpy(dtype=np.float64) - m["risk_lowi"].to_numpy(dtype=np.float64)
                dff = dff[np.isfinite(dff)]
                if np.any(dff != 0):
                    try:
                        diffp = (m["risk_auc"] - m["risk_lowi"]).to_numpy(dtype=np.float64)
                        diffp = diffp[np.isfinite(diffp)]
                        _, p_w = scipy_stats.wilcoxon(diffp, zero_method="wilcox", alternative="two-sided")
                        p_w = str(float(p_w))
                    except Exception:  # noqa: BLE001
                        p_w = ""
            stat_rows.append(
                {
                    "epsilon": eps,
                    "coverage": cov,
                    "metric": "wilcoxon_paired_risk_auc_vs_lowI",
                    "n": int(len(m)),
                    "mean": float(np.mean(m["risk_auc"] - m["risk_lowi"])) if len(m) else float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "p_value_if_available": p_w,
                    "warning": "",
                }
            )

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(os.path.join(out_dir, "statistical_stability_summary.csv"), index=False)

    plot_df = stat_df[
        stat_df["metric"].isin(["instability_separation", "mean_delta_risk_failure", "mean_gap_reduction_failure"])
        & (stat_df["epsilon"] == 0.002)
    ]
    if len(plot_df):
        fig, ax = plt.subplots(figsize=(9, 5))
        covs_u = sorted(plot_df["coverage"].unique())
        x = np.arange(len(covs_u))
        metrics = ["instability_separation", "mean_delta_risk_failure", "mean_gap_reduction_failure"]
        w = 0.25
        for i, met in enumerate(metrics):
            sub = plot_df[plot_df["metric"] == met].set_index("coverage").reindex(covs_u)
            means = sub["mean"].to_numpy(dtype=np.float64)
            lo = sub["ci_low"].to_numpy(dtype=np.float64)
            hi = sub["ci_high"].to_numpy(dtype=np.float64)
            yerr = np.vstack(
                [
                    np.nan_to_num(means - lo, nan=0.0, posinf=0.0, neginf=0.0),
                    np.nan_to_num(hi - means, nan=0.0, posinf=0.0, neginf=0.0),
                ]
            )
            ax.bar(x + (i - 1) * w, means, width=w, yerr=yerr, label=met, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in covs_u])
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_title("Bootstrap 95% CI (eps=0.002, setting resample)")
        ax.legend(fontsize=7)
        _savefig_both(fig, os.path.join(out_dir, "bootstrap_ci_main_metrics"))

    fs_path = os.path.join(failure_dir, "failure_summary.csv")
    if os.path.isfile(fs_path):
        fsum = pd.read_csv(fs_path)
        fig, ax = plt.subplots(figsize=(7, 5))
        for eps_v in sorted(fsum["epsilon"].unique()):
            sub = fsum[fsum["epsilon"] == eps_v].sort_values("coverage")
            ax.plot(sub["coverage"], sub["mean_delta_risk_all"], marker="o", label=f"all eps={eps_v}")
            ax.plot(sub["coverage"], sub["mean_delta_risk_failure"], marker="s", linestyle="--", label=f"failure eps={eps_v}")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("coverage")
        ax.set_ylabel("mean delta risk (risk_auc - risk_stable)")
        ax.legend(fontsize=8)
        ax.set_title("Conditional gain: all vs failure (from failure_summary)")
        _savefig_both(fig, os.path.join(out_dir, "conditional_gain"))
        fig, ax = plt.subplots(figsize=(7, 5))
        for eps_v in sorted(fsum["epsilon"].unique()):
            sub = fsum[fsum["epsilon"] == eps_v].sort_values("coverage")
            ax.plot(sub["coverage"], sub["mean_gap_reduction"], marker="o", label=f"eps={eps_v}")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("coverage")
        ax.set_ylabel("mean gap_reduction (failure rows)")
        ax.legend()
        ax.set_title("Oracle gap reduction (failure_summary)")
        _savefig_both(fig, os.path.join(out_dir, "oracle_gap_reduction"))

    pd.DataFrame(wlog.rows).to_csv(os.path.join(out_dir, "warnings.csv"), index=False)

    # README claim block (auto from tables)
    c1 = ""
    if len(summ1_df) and "error_gap_high_minus_low" in summ1_df.columns:
        gaps = summ1_df["error_gap_high_minus_low"].dropna()
        c1 = f"- **Claim 1 (margin control)**: error_gap_high_minus_low > 0 for all buckets: **{bool((gaps > 0).all())}** (values: {gaps.tolist()})."
    c2 = ""
    if not alt_summ_df.empty:
        inst_m = float(alt_summ_df.loc[alt_summ_df["signal"] == "instability", "mean_spearman"].iloc[0])
        gap_m = float(alt_summ_df.loc[alt_summ_df["signal"] == "gap", "mean_spearman"].iloc[0])
        c2 = f"- **Claim 2 (not just margin/gap)**: mean Spearman instability **{inst_m:.4f}** vs gap **{gap_m:.4f}** (higher is better for ranking-error prediction if positive and larger than baselines)."
    c3 = ""
    if not fc_df.empty:
        subi = fc_df[(fc_df["signal"] == "instability") & (fc_df["epsilon"] == 0.002) & (fc_df["coverage"].isin([0.5, 0.7, 0.9]))]
        if len(subi):
            c3 = f"- **Claim 3 (failure)**: instability direction_correct for cov 0.5/0.7/0.9 (eps=0.002): **{bool(subi['direction_correct'].all())}**."
    c4 = ""
    if not reg_df.empty:
        r0 = reg_df[(reg_df["regime"] == "zero_I") & (reg_df["epsilon"] == 0.002) & (reg_df["coverage"] == 0.7)]
        rh = reg_df[(reg_df["regime"] == "high_I") & (reg_df["epsilon"] == 0.002) & (reg_df["coverage"] == 0.7)]
        if len(r0) and len(rh):
            c4 = f"- **Claim 4 (regimes)**: failure_rate @cov0.7 zero_I **{float(r0['failure_rate'].iloc[0]):.4f}** vs high_I **{float(rh['failure_rate'].iloc[0]):.4f}**."
    c5 = ""
    if not stat_df.empty:
        subb = stat_df[(stat_df["epsilon"] == 0.002) & (stat_df["metric"] == "instability_separation") & (stat_df["coverage"] == 0.7)]
        if len(subb):
            row = subb.iloc[0]
            ci_ok = np.isfinite(row["ci_low"]) and float(row["ci_low"]) > 0
            c5 = f"- **Claim 5 (bootstrap)**: instability_separation @cov0.7 mean **{float(row['mean']):.4f}**, 95% CI [{row['ci_low']}, {row['ci_high']}]; CI>0: **{ci_ok}**."
    claim_block = "\n".join([x for x in [c1, c2, c3, c4, c5] if x])

    _write_readme(out_dir, alt_summ_df, summ1_df, fc_df, summ_sel, reg_df, stat_df, claim_block)
    print(f"Done. Outputs in {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-dir", type=str, default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/pilot_instability_selection")
    p.add_argument("--promptad-root", type=str, default="/home/zju/mywork/NeurIPS2026/PromptAD")
    p.add_argument("--out-dir", type=str, default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/promptad_strengthening")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
