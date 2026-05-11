"""
Paired-sample tables aligned with PromptAD exp5 logic.

`build_sample_instability_table`, `build_sample_ranking_error_table`, and
`run_instability_rejection_experiment` are copied from
PromptAD/utils/pairwise_instability.py and run_instability_experiments.py.

`build_padim_protocol_b_pairwise_table` adapts the pair construction so that
flip uses three marginal scores (s0,s1,s2) while z_final uses fused s_fused,
matching PaDiM Protocol B in padim_instability_protocol_ab.compute_instability_metrics.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def build_padim_protocol_b_pairwise_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    All anomaly–normal pairs with PaDiM Protocol B definitions:
      z_k = 1[s_k(a) > s_k(n)] for k in {0,1,2} (marginals)
      flip = 1 if z0,z1,z2 are not all identical
      z_final = 1[s_fused(a) > s_fused(n)]
      pair_margin_var = var([s0_a-s0_n, s1_a-s1_n, s2_a-s2_n], ddof=0)
    """
    need = {"image_path", "image_label", "s0", "s1", "s2", "s_fused"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    a_df = df[df["image_label"] == "anomaly"][
        ["image_path", "s0", "s1", "s2", "s_fused"]
    ].copy()
    n_df = df[df["image_label"] == "normal"][
        ["image_path", "s0", "s1", "s2", "s_fused"]
    ].copy()
    if len(a_df) == 0 or len(n_df) == 0:
        raise ValueError("Need both anomaly and normal samples.")

    a = a_df.rename(
        columns={
            "image_path": "anomaly_path",
            "s0": "s0_a",
            "s1": "s1_a",
            "s2": "s2_a",
            "s_fused": "s_fused_a",
        }
    ).assign(_k=1)
    n = n_df.rename(
        columns={
            "image_path": "normal_path",
            "s0": "s0_n",
            "s1": "s1_n",
            "s2": "s2_n",
            "s_fused": "s_fused_n",
        }
    ).assign(_k=1)
    p = a.merge(n, on="_k", how="inner").drop(columns=["_k"])

    p["z0"] = (p["s0_a"] > p["s0_n"]).astype(np.int8)
    p["z1"] = (p["s1_a"] > p["s1_n"]).astype(np.int8)
    p["z2"] = (p["s2_a"] > p["s2_n"]).astype(np.int8)
    p["z_final"] = (p["s_fused_a"] > p["s_fused_n"]).astype(np.int8)
    all_same = (p["z0"] == p["z1"]) & (p["z1"] == p["z2"])
    p["flip"] = (~all_same).astype(np.int8)

    m0 = (p["s0_a"] - p["s0_n"]).to_numpy(dtype=float)
    m1 = (p["s1_a"] - p["s1_n"]).to_numpy(dtype=float)
    m2 = (p["s2_a"] - p["s2_n"]).to_numpy(dtype=float)
    margins = np.stack([m0, m1, m2], axis=1)
    p["pair_margin_var"] = np.var(margins, axis=1, ddof=0)

    return p[
        [
            "anomaly_path",
            "normal_path",
            "flip",
            "z_final",
            "pair_margin_var",
        ]
    ].copy()


def build_sample_instability_table(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df is None or not isinstance(pair_df, pd.DataFrame):
        raise ValueError("pair_df must be a pandas DataFrame")
    if len(pair_df) == 0:
        raise ValueError("pair_df is empty; cannot build sample instability table.")

    required = {"anomaly_path", "normal_path", "flip", "pair_margin_var"}
    missing = sorted(list(required - set(pair_df.columns)))
    if missing:
        raise ValueError(f"pair_df missing required columns: {missing}")

    anomaly_agg = (
        pair_df.groupby("anomaly_path", as_index=False)
        .agg(
            I_bin=("flip", "mean"),
            I_cont=("pair_margin_var", "mean"),
            num_pairs=("flip", "size"),
        )
        .rename(columns={"anomaly_path": "image_path"})
    )
    anomaly_agg["image_label"] = "anomaly"

    normal_agg = (
        pair_df.groupby("normal_path", as_index=False)
        .agg(
            I_bin=("flip", "mean"),
            I_cont=("pair_margin_var", "mean"),
            num_pairs=("flip", "size"),
        )
        .rename(columns={"normal_path": "image_path"})
    )
    normal_agg["image_label"] = "normal"

    sample_df = pd.concat([anomaly_agg, normal_agg], ignore_index=True)
    sample_df = sample_df[["image_path", "image_label", "I_bin", "I_cont", "num_pairs"]]

    return sample_df


def build_sample_ranking_error_table(
    pair_df: pd.DataFrame,
    sample_instability_df: pd.DataFrame,
    per_sample_df: pd.DataFrame,
) -> pd.DataFrame:
    need = {"anomaly_path", "normal_path", "z_final"}
    if not need.issubset(pair_df.columns):
        raise ValueError(f"pair_df missing columns: {sorted(need - set(pair_df.columns))}")

    def _mean_error(s: pd.Series) -> float:
        return float(1.0 - np.mean(s.to_numpy(dtype=float)))

    err_anom = (
        pair_df.groupby("anomaly_path", as_index=False)
        .agg(error=("z_final", _mean_error), num_pairs=("z_final", "size"))
        .rename(columns={"anomaly_path": "image_path"})
    )
    err_anom["image_label"] = "anomaly"

    err_norm = (
        pair_df.groupby("normal_path", as_index=False)
        .agg(error=("z_final", _mean_error), num_pairs=("z_final", "size"))
        .rename(columns={"normal_path": "image_path"})
    )
    err_norm["image_label"] = "normal"

    err_tbl = pd.concat([err_anom, err_norm], ignore_index=True)

    si_full = sample_instability_df[["image_path", "image_label", "I_bin"]].drop_duplicates(
        subset=["image_path"]
    )
    merged = err_tbl.merge(si_full, on="image_path", how="inner", validate="one_to_one", suffixes=("_pair", "_si"))
    if not (merged["image_label_pair"] == merged["image_label_si"]).all():
        raise ValueError("image_label mismatch between pair-derived table and sample_instability_table.")
    merged["image_label"] = merged["image_label_pair"]
    merged = merged[["image_path", "image_label", "I_bin", "error", "num_pairs"]]

    per = per_sample_df[["image_path", "harmonic_score"]].drop_duplicates(subset=["image_path"])
    merged = merged.merge(per, on="image_path", how="inner", validate="one_to_one")

    return merged[
        ["image_path", "image_label", "I_bin", "harmonic_score", "error", "num_pairs"]
    ].copy()


def _rejection_split_metrics(remaining: pd.DataFrame) -> Dict[str, Any]:
    if len(remaining) == 0:
        return {
            "remaining_sample_count": 0,
            "remaining_anomaly_count": 0,
            "remaining_normal_count": 0,
            "mean_error_all": float("nan"),
            "mean_error_anomaly": float("nan"),
            "mean_error_normal": float("nan"),
            "mean_harmonic_score_normal": float("nan"),
            "mean_harmonic_score_anomaly": float("nan"),
        }

    is_a = remaining["image_label"] == "anomaly"
    is_n = remaining["image_label"] == "normal"
    return {
        "remaining_sample_count": int(len(remaining)),
        "remaining_anomaly_count": int(is_a.sum()),
        "remaining_normal_count": int(is_n.sum()),
        "mean_error_all": float(remaining["error"].mean()),
        "mean_error_anomaly": float(remaining.loc[is_a, "error"].mean()) if is_a.any() else float("nan"),
        "mean_error_normal": float(remaining.loc[is_n, "error"].mean()) if is_n.any() else float("nan"),
        "mean_harmonic_score_normal": float(remaining.loc[is_n, "harmonic_score"].mean())
        if is_n.any()
        else float("nan"),
        "mean_harmonic_score_anomaly": float(remaining.loc[is_a, "harmonic_score"].mean())
        if is_a.any()
        else float("nan"),
    }


def run_instability_rejection_experiment(
    pair_df: pd.DataFrame,
    sample_instability_path: str,
    per_sample_csv_path: str,
    out_dir: str,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    si = pd.read_csv(sample_instability_path)
    per = pd.read_csv(per_sample_csv_path)

    sample_rank_df = build_sample_ranking_error_table(pair_df, si, per)
    p_sample = os.path.join(out_dir, "exp5_sample_ranking_error.csv")
    sample_rank_df.to_csv(p_sample, index=False)

    rows: List[Dict[str, Any]] = []
    rows.append({"setting": "baseline", **_rejection_split_metrics(sample_rank_df)})

    n = len(sample_rank_df)
    sorted_df = sample_rank_df.sort_values("I_bin", ascending=False, kind="mergesort").reset_index(drop=True)

    for pct, name in [(0.10, "reject_10"), (0.20, "reject_20"), (0.30, "reject_30")]:
        k_reject = int(np.ceil(n * pct))
        k_reject = min(k_reject, n)
        remaining = sorted_df.iloc[k_reject:]
        rows.append({"setting": name, **_rejection_split_metrics(remaining)})

    out_df = pd.DataFrame(rows)
    col_order = [
        "setting",
        "remaining_sample_count",
        "remaining_anomaly_count",
        "remaining_normal_count",
        "mean_error_all",
        "mean_error_anomaly",
        "mean_error_normal",
        "mean_harmonic_score_normal",
        "mean_harmonic_score_anomaly",
    ]
    out_df = out_df[[c for c in col_order if c in out_df.columns]]
    p_rej = os.path.join(out_dir, "exp5_instability_rejection.csv")
    out_df.to_csv(p_rej, index=False)
    return p_sample, p_rej
