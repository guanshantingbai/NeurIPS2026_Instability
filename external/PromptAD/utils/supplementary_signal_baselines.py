#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pilot_instability_aware_selection import (
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


def _rank01(x: np.ndarray) -> np.ndarray:
    if x.size <= 1:
        return np.zeros_like(x, dtype=np.float64)
    r = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    return (r - 1.0) / (float(x.size) - 1.0)


def _build_seed_agg(promptad_root: Path) -> pd.DataFrame:
    seed_rows: List[Dict[str, Any]] = []
    seen = set()
    for path in _discover_per_sample_csvs(str(promptad_root)):
        parsed = _parse_setting_seed_from_path(path)
        if parsed is None:
            continue
        key = (parsed.dataset, parsed.cls, parsed.k, parsed.seed)
        if key in seen:
            continue
        seen.add(key)
        try:
            raw = pd.read_csv(path)
            w = _load_standardized_per_sample(raw)
        except Exception:
            continue
        if len(w) < 2:
            continue
        sem = w["semantic_score"].to_numpy(dtype=np.float64)
        vis = w["visual_score"].to_numpy(dtype=np.float64)
        fin = w["final_score"].to_numpy(dtype=np.float64)
        y = w["image_label"].to_numpy(dtype=np.int8)

        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        sem_pos, sem_neg = sem[pos_idx], sem[neg_idx]
        vis_pos, vis_neg = vis[pos_idx], vis[neg_idx]
        fp, fn = fin[pos_idx], fin[neg_idx]
        z_sem = sem_pos[:, None] > sem_neg[None, :]
        z_vis = vis_pos[:, None] > vis_neg[None, :]
        flip_m = (z_sem != z_vis).astype(np.float64)
        margin_m = (fp[:, None] - fn[None, :]).astype(np.float64)

        gap = np.abs(sem - vis)
        m2 = 0.5 * (sem + vis)
        score_var = 0.5 * ((sem - m2) ** 2 + (vis - m2) ** 2)
        mean_abs_margin_s = np.zeros(len(w), dtype=np.float64)
        mean_abs_margin_s[pos_idx] = np.mean(np.abs(margin_m), axis=1)
        mean_abs_margin_s[neg_idx] = np.mean(np.abs(margin_m), axis=0)

        # Baseline A: global ranking disagreement (normalized to [0,1])
        # kendall_disagreement = (1 - tau) / 2
        tau = pd.Series(sem).corr(pd.Series(vis), method="kendall")
        tau = float(tau) if pd.notna(tau) else 0.0
        kendall_disagreement = (1.0 - tau) / 2.0

        # Baseline B: branch rank disagreement (view-level inconsistency)
        sem_r = _rank01(sem)
        vis_r = _rank01(vis)
        branch_disagreement = float(np.mean(np.abs(sem_r - vis_r)))

        seed_rows.append(
            {
                "dataset": parsed.dataset,
                "class": parsed.cls,
                "k": int(parsed.k),
                "seed_auc": int(parsed.seed),
                "gap_setting": float(np.mean(gap)),
                "score_var_setting": float(np.mean(score_var)),
                "margin_setting": float(np.mean(mean_abs_margin_s)),
                "kendall_disagreement_setting": float(kendall_disagreement),
                "branch_disagreement_setting": float(branch_disagreement),
            }
        )
    return pd.DataFrame(seed_rows)


def _load_failure_with_seed(pilot_dir: Path) -> pd.DataFrame:
    fa = pd.read_csv(pilot_dir / "failure_driven" / "failure_analysis.csv")
    if "seed_auc" in fa.columns and not fa["seed_auc"].isna().all():
        return fa
    sel_list = []
    for eps in [0.002, 0.005]:
        p = pilot_dir / f"selection_summary_epsilon_{eps:.3f}.csv"
        if p.is_file():
            sel_list.append(pd.read_csv(p))
    if not sel_list:
        raise FileNotFoundError("selection_summary_epsilon_*.csv not found")
    sel = pd.concat(sel_list, axis=0, ignore_index=True)
    key = ["dataset", "class", "k", "coverage", "epsilon", "auroc_auc_seed"]
    sel = sel[[c for c in key + ["seed_auc"] if c in sel.columns]].drop_duplicates()
    merged = fa.merge(sel, on=[c for c in key if c in fa.columns and c in sel.columns], how="left")
    return merged


def run(args: argparse.Namespace) -> None:
    promptad_root = Path(args.promptad_root).resolve()
    pilot_dir = Path(args.pilot_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_agg = _build_seed_agg(promptad_root)
    seed_agg.to_csv(out_dir / "signal_baselines_seed_agg.csv", index=False)

    fa = _load_failure_with_seed(pilot_dir)
    fa = fa.merge(seed_agg, on=["dataset", "class", "k", "seed_auc"], how="left")

    signals = [
        ("instability", "instability_auc_seed"),
        ("kendall_disagreement", "kendall_disagreement_setting"),
        ("branch_disagreement", "branch_disagreement_setting"),
        ("margin", "margin_setting"),
        ("gap", "gap_setting"),
        ("score_var", "score_var_setting"),
    ]
    rows: List[Dict[str, Any]] = []
    for (eps, cov), g in fa.groupby(["epsilon", "coverage"], sort=True):
        gf = g[g["failure"] == True]
        gn = g[g["failure"] == False]
        for name, col in signals:
            vf = pd.to_numeric(gf[col], errors="coerce").dropna()
            vn = pd.to_numeric(gn[col], errors="coerce").dropna()
            mf = float(vf.mean()) if len(vf) else float("nan")
            mn = float(vn.mean()) if len(vn) else float("nan")
            rows.append(
                {
                    "epsilon": float(eps),
                    "coverage": float(cov),
                    "signal": name,
                    "n_failure": int(len(gf)),
                    "n_non_failure": int(len(gn)),
                    "mean_failure": mf,
                    "mean_non_failure": mn,
                    "difference": mf - mn if np.isfinite(mf) and np.isfinite(mn) else float("nan"),
                    "ratio": mf / mn if np.isfinite(mf) and np.isfinite(mn) and mn != 0 else float("nan"),
                }
            )
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "failure_conditioned_signal_with_baselines.csv", index=False)

    order = ["instability", "kendall_disagreement", "branch_disagreement", "margin", "gap", "score_var"]
    vis = comp[comp["coverage"].isin([0.5, 0.7, 0.8])].copy()
    summ = vis.groupby("signal", as_index=False)[["mean_failure", "mean_non_failure", "difference"]].mean()
    summ["signal"] = pd.Categorical(summ["signal"], categories=order, ordered=True)
    summ = summ.sort_values("signal").reset_index(drop=True)
    summ["ratio"] = summ["mean_failure"] / summ["mean_non_failure"]
    summ.to_csv(out_dir / "failure_signal_baseline_summary.csv", index=False)

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
    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    x = np.arange(len(order))
    w = 0.36
    ax.bar(x - w / 2, summ["mean_failure"], width=w, color="#1f77b4", label="failure")
    ax.bar(x + w / 2, summ["mean_non_failure"], width=w, color="#ff7f0e", label="non-failure")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("Mean signal value")
    ax.set_title("Failure vs non-failure signal comparison (with additional baselines)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "failure_signal_comparison_with_baselines.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "failure_signal_comparison_with_baselines.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote: {out_dir}")
    print(summ.to_string(index=False))


def _default_repo_root() -> Path:
    # external/PromptAD/utils/this_file -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    repo = _default_repo_root()
    default_promptad = repo / "src/experiments/app_signal_comparison/samples/fastpath/promptad_stub"
    default_pilot = repo / "src/experiments/app_signal_comparison/samples/fastpath/pilot_instability_selection"
    default_out = repo / "outputs/figures/app_signal_comparison"
    p = argparse.ArgumentParser()
    p.add_argument("--promptad-root", type=Path, default=default_promptad)
    p.add_argument("--pilot-dir", type=Path, default=default_pilot)
    p.add_argument("--out-dir", type=Path, default=default_out)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
