#!/usr/bin/env python3
"""
Section 4.2 same-AUROC controlled analysis.

Given a per-seed setting CSV, build:
  A) same-setting seed pairs (group key: dataset, class, k)
  B) all-setting pairs (all rows, i<j)

For each pair type and epsilon in {0.002, 0.005}, filter by:
  delta_auroc < epsilon
and report summary statistics on delta_instability.
"""

from __future__ import annotations

import argparse
import itertools
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

AUROC_CANDIDATES = ["auroc", "i_roc", "image_auroc", "sklearn_auroc_final"]
INSTABILITY_CANDIDATES = [
    "instability",
    "flip_rate",
    "mean_instability",
    "flip_rate_mean",
    "mean_I_bin",
]


def _pick_column(df: pd.DataFrame, candidates: Iterable[str], role: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find {role} column in {list(candidates)}. Present: {list(df.columns)}")


def load_and_standardize(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    need_base = ["dataset", "seed"]
    for c in need_base:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if "class" not in df.columns and "category" in df.columns:
        df = df.rename(columns={"category": "class"})
    if "class" not in df.columns:
        raise ValueError("Missing class/category column.")
    if "k" not in df.columns:
        raise ValueError("Missing k column.")

    auroc_col = _pick_column(df, AUROC_CANDIDATES, "auroc")
    instability_col = _pick_column(df, INSTABILITY_CANDIDATES, "instability")
    if auroc_col != "auroc":
        df = df.rename(columns={auroc_col: "auroc"})
    if instability_col != "instability":
        df = df.rename(columns={instability_col: "instability"})

    core = df[["dataset", "class", "k", "seed", "auroc", "instability"]].copy()
    core["dataset"] = core["dataset"].astype(str)
    core["class"] = core["class"].astype(str)
    core["k"] = pd.to_numeric(core["k"], errors="coerce")
    core["seed"] = pd.to_numeric(core["seed"], errors="coerce")
    core["auroc"] = pd.to_numeric(core["auroc"], errors="coerce")
    core["instability"] = pd.to_numeric(core["instability"], errors="coerce")
    core = core.dropna(subset=["k", "seed", "auroc", "instability"]).copy()
    core["k"] = core["k"].astype(int)
    core["seed"] = core["seed"].astype(int)
    core = core.drop_duplicates(subset=["dataset", "class", "k", "seed"], keep="first")
    return core


def _pair_row(ri: pd.Series, rj: pd.Series) -> Dict[str, object]:
    da = abs(float(ri["auroc"]) - float(rj["auroc"]))
    di = abs(float(ri["instability"]) - float(rj["instability"]))
    return {
        "dataset_i": ri["dataset"],
        "class_i": ri["class"],
        "k_i": int(ri["k"]),
        "seed_i": int(ri["seed"]),
        "dataset_j": rj["dataset"],
        "class_j": rj["class"],
        "k_j": int(rj["k"]),
        "seed_j": int(rj["seed"]),
        "auroc_i": float(ri["auroc"]),
        "auroc_j": float(rj["auroc"]),
        "instability_i": float(ri["instability"]),
        "instability_j": float(rj["instability"]),
        "delta_auroc": float(da),
        "delta_instability": float(di),
    }


def build_same_setting_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, g in df.groupby(["dataset", "class", "k"], sort=False):
        if len(g) < 2:
            continue
        records = list(g.to_dict(orient="records"))
        for i, j in itertools.combinations(range(len(records)), 2):
            rows.append(_pair_row(pd.Series(records[i]), pd.Series(records[j])))
    return pd.DataFrame(rows)


def build_all_setting_pairs(df: pd.DataFrame) -> pd.DataFrame:
    recs = list(df.to_dict(orient="records"))
    rows: List[Dict[str, object]] = []
    for i, j in itertools.combinations(range(len(recs)), 2):
        rows.append(_pair_row(pd.Series(recs[i]), pd.Series(recs[j])))
    return pd.DataFrame(rows)


def summary_stats(pair_df: pd.DataFrame, pair_type: str, epsilon: float) -> Dict[str, object]:
    filt = pair_df[pair_df["delta_auroc"] < float(epsilon)].copy()
    if filt.empty:
        return {
            "pair_type": pair_type,
            "epsilon": float(epsilon),
            "number_of_pairs": 0,
            "median_delta_instability": np.nan,
            "mean_delta_instability": np.nan,
            "p90_delta_instability": np.nan,
            "max_delta_instability": np.nan,
        }
    d = filt["delta_instability"].to_numpy(dtype=float)
    return {
        "pair_type": pair_type,
        "epsilon": float(epsilon),
        "number_of_pairs": int(len(filt)),
        "median_delta_instability": float(np.median(d)),
        "mean_delta_instability": float(np.mean(d)),
        "p90_delta_instability": float(np.percentile(d, 90)),
        "max_delta_instability": float(np.max(d)),
    }


def write_outputs(
    same_df: pd.DataFrame,
    all_df: pd.DataFrame,
    out_dir: str,
    epsilons: Tuple[float, float],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    e1, e2 = epsilons
    e1_tag = f"{int(round(e1 * 10000)):04d}"
    e2_tag = f"{int(round(e2 * 10000)):04d}"

    summary = pd.DataFrame(
        [
            summary_stats(same_df, "same_setting", e1),
            summary_stats(all_df, "all_setting", e1),
            summary_stats(same_df, "same_setting", e2),
            summary_stats(all_df, "all_setting", e2),
        ]
    )
    summary.to_csv(os.path.join(out_dir, "summary_same_auroc.csv"), index=False)

    same_e1 = same_df[same_df["delta_auroc"] < e1].copy()
    all_e1 = all_df[all_df["delta_auroc"] < e1].copy()
    same_e2 = same_df[same_df["delta_auroc"] < e2].copy()
    all_e2 = all_df[all_df["delta_auroc"] < e2].copy()

    same_e1.to_csv(os.path.join(out_dir, f"pairs_same_setting_eps{e1_tag}.csv"), index=False)
    all_e1.to_csv(os.path.join(out_dir, f"pairs_all_setting_eps{e1_tag}.csv"), index=False)
    same_e2.to_csv(os.path.join(out_dir, f"pairs_same_setting_eps{e2_tag}.csv"), index=False)
    all_e2.to_csv(os.path.join(out_dir, f"pairs_all_setting_eps{e2_tag}.csv"), index=False)

    # Top-10 by delta_instability under each (pair_type, epsilon)
    same_e1.sort_values(["delta_instability", "delta_auroc"], ascending=[False, True]).head(10).to_csv(
        os.path.join(out_dir, f"top10_same_setting_eps{e1_tag}.csv"), index=False
    )
    all_e1.sort_values(["delta_instability", "delta_auroc"], ascending=[False, True]).head(10).to_csv(
        os.path.join(out_dir, f"top10_all_setting_eps{e1_tag}.csv"), index=False
    )
    same_e2.sort_values(["delta_instability", "delta_auroc"], ascending=[False, True]).head(10).to_csv(
        os.path.join(out_dir, f"top10_same_setting_eps{e2_tag}.csv"), index=False
    )
    all_e2.sort_values(["delta_instability", "delta_auroc"], ascending=[False, True]).head(10).to_csv(
        os.path.join(out_dir, f"top10_all_setting_eps{e2_tag}.csv"), index=False
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        type=str,
        default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/per_seed_setting_metrics.csv",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/section42_same_auroc",
    )
    p.add_argument("--eps1", type=float, default=0.002)
    p.add_argument("--eps2", type=float, default=0.005)
    args = p.parse_args()

    df = load_and_standardize(args.input_csv)
    same_df = build_same_setting_pairs(df)
    all_df = build_all_setting_pairs(df)
    write_outputs(same_df, all_df, args.out_dir, (float(args.eps1), float(args.eps2)))

    print(f"input_rows={len(df)}")
    print(f"same_setting_pairs={len(same_df)}")
    print(f"all_setting_pairs={len(all_df)}")
    print(f"wrote={os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
