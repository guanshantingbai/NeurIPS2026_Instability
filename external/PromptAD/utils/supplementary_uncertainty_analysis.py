#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_failure_with_seed(pilot_dir: Path) -> pd.DataFrame:
    fa = pd.read_csv(pilot_dir / "failure_driven" / "failure_analysis.csv")
    sel_list = []
    for eps in [0.002, 0.005]:
        p = pilot_dir / f"selection_summary_epsilon_{eps:.3f}.csv"
        if p.is_file():
            sel_list.append(pd.read_csv(p))
    if not sel_list:
        raise FileNotFoundError("selection_summary_epsilon_*.csv not found")
    sel = pd.concat(sel_list, axis=0, ignore_index=True)
    # Recover seed ids used by AUROC and oracle selections for each candidate pair.
    key_cols = ["dataset", "class", "k", "coverage", "epsilon", "auroc_auc_seed", "seed_auc", "seed_oracle"]
    sel = sel[[c for c in key_cols if c in sel.columns]].drop_duplicates()
    on = [c for c in ["dataset", "class", "k", "coverage", "epsilon", "auroc_auc_seed"] if c in fa.columns and c in sel.columns]
    return fa.merge(sel, on=on, how="left")


def _build_candidate_level_df(repo_root: Path) -> pd.DataFrame:
    pilot = repo_root / "PromptAD" / "result_analysis" / "pilot_instability_selection"
    fa = _load_failure_with_seed(pilot)

    # Oracle instability from per_seed metrics
    per = pd.read_csv(pilot / "per_seed_metrics.csv")
    inst_col = "setting_instability_flip" if "setting_instability_flip" in per.columns else "setting_instability"
    oracle = per[["dataset", "class", "k", "seed", inst_col]].rename(
        columns={"seed": "seed_oracle", inst_col: "instability_oracle_seed"}
    )
    fa = fa.merge(oracle, on=["dataset", "class", "k", "seed_oracle"], how="left")

    # Additional baseline signals computed previously at (dataset,class,k,seed_auc)
    seed_agg = pd.read_csv(
        repo_root / "paper_figures" / "promptad" / "supplementary" / "signal_baselines_seed_agg.csv"
    )
    fa = fa.merge(seed_agg, on=["dataset", "class", "k", "seed_auc"], how="left")

    fa["risk_gap"] = pd.to_numeric(fa["risk_auc_seed"], errors="coerce") - pd.to_numeric(fa["risk_oracle_seed"], errors="coerce")
    fa["instability_gap"] = pd.to_numeric(fa["instability_auc_seed"], errors="coerce") - pd.to_numeric(
        fa["instability_oracle_seed"], errors="coerce"
    )
    return fa


def _bootstrap_mean_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = x[idx]
    means = samples.mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def _bootstrap_sep_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> Tuple[float, float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    na, nb = a.size, b.size
    ia = rng.integers(0, na, size=(n_boot, na))
    ib = rng.integers(0, nb, size=(n_boot, nb))
    sep = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(sep, [0.025, 0.975])
    point = float(a.mean() - b.mean())
    return point, float(lo), float(hi)


def _wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n) / denom
    return float(phat), float(center - half), float(center + half)


def run(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / "paper_figures" / "promptad" / "supplementary").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = _build_candidate_level_df(repo_root)

    # Task A: failure statistics CIs
    rows_a: List[Dict[str, object]] = []
    for subset_name, sub in [("all", df), ("failure-only", df[df["failure"] == True])]:
        for metric, col in [("mean risk gap", "risk_gap"), ("mean instability gap", "instability_gap")]:
            point, lo, hi = _bootstrap_mean_ci(pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64), args.n_boot, rng)
            rows_a.append(
                {
                    "subset": subset_name,
                    "metric": metric,
                    "mean": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_pairs": int(len(sub)),
                }
            )
    ta = pd.DataFrame(rows_a)
    ta.to_csv(out_dir / "bootstrap_failure_statistics_ci.csv", index=False)

    # Task B: signal separation CIs
    signal_cols = {
        "instability": "instability_auc_seed",
        "kendall_disagreement": "kendall_disagreement_setting",
        "branch_disagreement": "branch_disagreement_setting",
        "margin": "margin_setting",
        "gap": "gap_setting",
        "score_var": "score_var_setting",
    }
    rows_b: List[Dict[str, object]] = []
    df_f = df[df["failure"] == True]
    df_nf = df[df["failure"] == False]
    for sig, col in signal_cols.items():
        a = pd.to_numeric(df_f[col], errors="coerce").to_numpy(dtype=np.float64)
        b = pd.to_numeric(df_nf[col], errors="coerce").to_numpy(dtype=np.float64)
        point, lo, hi = _bootstrap_sep_ci(a, b, args.n_boot, rng)
        rows_b.append(
            {
                "signal": sig,
                "separation_failure_minus_nonfailure": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_failure": int(np.isfinite(a).sum()),
                "n_non_failure": int(np.isfinite(b).sum()),
                "ci_crosses_zero": bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi),
            }
        )
    tb = pd.DataFrame(rows_b).sort_values("signal").reset_index(drop=True)
    tb.to_csv(out_dir / "bootstrap_signal_separation_ci.csv", index=False)

    # Task C: binomial CI
    phat, lo, hi = _wilson_interval(args.k_success, args.n_total)
    tc = pd.DataFrame(
        [
            {
                "k_success": int(args.k_success),
                "n_total": int(args.n_total),
                "proportion": phat,
                "ci95_low": lo,
                "ci95_high": hi,
                "method": "wilson",
            }
        ]
    )
    tc.to_csv(out_dir / "binomial_generalization_ci.csv", index=False)

    print(f"wrote: {out_dir / 'bootstrap_failure_statistics_ci.csv'}")
    print(f"wrote: {out_dir / 'bootstrap_signal_separation_ci.csv'}")
    print(f"wrote: {out_dir / 'binomial_generalization_ci.csv'}")
    print("\n[Task A]")
    print(ta.to_string(index=False))
    print("\n[Task B]")
    print(tb.to_string(index=False))
    print("\n[Task C]")
    print(tc.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default="/home/zju/mywork/NeurIPS2026")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-success", type=int, default=69)
    p.add_argument("--n-total", type=int, default=72)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
