import os
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REQUIRED_COLUMNS: List[str] = [
    "image_path",
    "image_label",
    "semantic_score",
    "visual_score",
    "harmonic_score",
]

ALLOWED_LABELS = {"anomaly", "normal"}


def load_score_table(csv_path: str) -> pd.DataFrame:
    """
    Load a sample-level score table from CSV and validate schema.

    Required columns:
      ["image_path", "image_label", "semantic_score", "visual_score", "harmonic_score"]

    image_label must be in {"anomaly", "normal"}.
    """
    if not csv_path or not isinstance(csv_path, str):
        raise ValueError("csv_path must be a non-empty string")
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got columns: {list(df.columns)}")

    bad_labels = set(df["image_label"].dropna().unique()) - ALLOWED_LABELS
    if bad_labels:
        raise ValueError(f"image_label contains invalid values: {sorted(bad_labels)}. Allowed: {sorted(ALLOWED_LABELS)}")

    # Ensure numeric columns are numeric (raise if conversion fails)
    for col in ["semantic_score", "visual_score", "harmonic_score"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["image_path"] = df["image_path"].astype(str)
    df["image_label"] = df["image_label"].astype(str)

    return df


def build_pairwise_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all anomaly-normal pairs and compute pair-level metrics.

    Only cross-label pairs are constructed:
      anomaly sample x_i^+ paired with normal sample x_j^-.

    Definitions (do NOT change):
      z_sem   = 1[sem_a > sem_n] else 0
      z_vis   = 1[vis_a > vis_n] else 0
      z_final = 1[final_a > final_n] else 0

      m_sem   = sem_a - sem_n
      m_vis   = vis_a - vis_n
      m_final = final_a - final_n

      flip = 1 if z_sem, z_vis, z_final are NOT all identical; else 0

      pair_margin_var = var([m_sem, m_vis, m_final])   # numpy.var, ddof=0
      pair_conflict   = abs(m_sem - m_vis)
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    anomaly_df = df[df["image_label"] == "anomaly"][["image_path", "semantic_score", "visual_score", "harmonic_score"]].copy()
    normal_df = df[df["image_label"] == "normal"][["image_path", "semantic_score", "visual_score", "harmonic_score"]].copy()

    if len(anomaly_df) == 0:
        raise ValueError("No anomaly samples found; cannot build anomaly-normal pairs.")
    if len(normal_df) == 0:
        raise ValueError("No normal samples found; cannot build anomaly-normal pairs.")

    # Cartesian product via merge on a constant key (vectorized; no row-wise append).
    a = anomaly_df.rename(
        columns={
            "image_path": "anomaly_path",
            "semantic_score": "sem_a",
            "visual_score": "vis_a",
            "harmonic_score": "final_a",
        }
    ).assign(_k=1)
    n = normal_df.rename(
        columns={
            "image_path": "normal_path",
            "semantic_score": "sem_n",
            "visual_score": "vis_n",
            "harmonic_score": "final_n",
        }
    ).assign(_k=1)

    pair_df = a.merge(n, on="_k", how="inner").drop(columns=["_k"])

    # Pairwise correctness (binary)
    # z_* = 1[ score(anomaly) > score(normal) ] else 0
    pair_df["z_sem"] = (pair_df["sem_a"] > pair_df["sem_n"]).astype(np.int8)
    pair_df["z_vis"] = (pair_df["vis_a"] > pair_df["vis_n"]).astype(np.int8)
    pair_df["z_final"] = (pair_df["final_a"] > pair_df["final_n"]).astype(np.int8)

    # Pairwise margin (continuous)
    pair_df["m_sem"] = pair_df["sem_a"] - pair_df["sem_n"]
    pair_df["m_vis"] = pair_df["vis_a"] - pair_df["vis_n"]
    pair_df["m_final"] = pair_df["final_a"] - pair_df["final_n"]

    # flip: 1 if z_sem, z_vis, z_final not all the same, else 0
    all_same = (pair_df["z_sem"] == pair_df["z_vis"]) & (pair_df["z_vis"] == pair_df["z_final"])
    pair_df["flip"] = (~all_same).astype(np.int8)

    # pair_margin_var: variance([m_sem, m_vis, m_final]) with ddof=0
    margins = np.stack(
        [
            pair_df["m_sem"].to_numpy(dtype=float),
            pair_df["m_vis"].to_numpy(dtype=float),
            pair_df["m_final"].to_numpy(dtype=float),
        ],
        axis=1,
    )
    pair_df["pair_margin_var"] = np.var(margins, axis=1, ddof=0)

    # pair_conflict: abs(m_sem - m_vis)
    pair_df["pair_conflict"] = np.abs(pair_df["m_sem"] - pair_df["m_vis"])

    return pair_df


def compute_pairwise_auroc_from_table(pair_df: pd.DataFrame) -> float:
    """
    Discrete AUROC defined from pairwise correctness:

      AUROC_pair = mean(z_final)

    where z_final = 1[final_a > final_n] for each anomaly-normal pair.
    Returns a float in [0, 1].
    """
    if pair_df is None or not isinstance(pair_df, pd.DataFrame):
        raise ValueError("pair_df must be a pandas DataFrame")
    if "z_final" not in pair_df.columns:
        raise ValueError("pair_df must contain column 'z_final'")
    if len(pair_df) == 0:
        raise ValueError("pair_df is empty; cannot compute pairwise AUROC.")
    return float(pair_df["z_final"].mean())


def compute_sklearn_auroc(df: pd.DataFrame, score_col: str = "harmonic_score") -> float:
    """
    Standard AUROC using sklearn.metrics.roc_auc_score.

    Labels:
      anomaly -> 1
      normal  -> 0
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if score_col not in df.columns:
        raise ValueError(f"score_col not found: {score_col}")

    y = df["image_label"].map({"normal": 0, "anomaly": 1})
    if y.isna().any():
        raise ValueError("Found invalid labels when mapping image_label to {normal:0, anomaly:1}.")

    if y.nunique() < 2:
        raise ValueError("AUROC is undefined when only one class is present in labels.")

    scores = pd.to_numeric(df[score_col], errors="raise")
    return float(roc_auc_score(y.to_numpy(dtype=int), scores.to_numpy(dtype=float)))


def build_sample_instability_table(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pair-level quantities into sample-level instability metrics.

    For anomaly sample x_i^+:
      I_bin(x_i^+)  = mean over all normal samples of flip(i,j)
      I_cont(x_i^+) = mean over all normal samples of pair_margin_var(i,j)

    For normal sample x_j^-:
      I_bin(x_j^-)  = mean over all anomaly samples of flip(i,j)
      I_cont(x_j^-) = mean over all anomaly samples of pair_margin_var(i,j)
    """
    if pair_df is None or not isinstance(pair_df, pd.DataFrame):
        raise ValueError("pair_df must be a pandas DataFrame")
    if len(pair_df) == 0:
        raise ValueError("pair_df is empty; cannot build sample instability table.")

    required = {"anomaly_path", "normal_path", "flip", "pair_margin_var"}
    missing = sorted(list(required - set(pair_df.columns)))
    if missing:
        raise ValueError(f"pair_df missing required columns: {missing}")

    # For anomaly samples: aggregate over all normals they pair with
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

    # For normal samples: aggregate over all anomalies they pair with
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


def summarize_instability_results(
    df: pd.DataFrame,
    pair_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Summarize key statistics into a dict.
    """
    if df is None or pair_df is None or sample_df is None:
        raise ValueError("df, pair_df, sample_df must be provided")

    sklearn_auroc_final = compute_sklearn_auroc(df, score_col="harmonic_score")
    pairwise_auroc_final = compute_pairwise_auroc_from_table(pair_df)

    summary: Dict[str, Any] = {
        "num_samples": int(len(df)),
        "num_anomaly": int((df["image_label"] == "anomaly").sum()),
        "num_normal": int((df["image_label"] == "normal").sum()),
        "num_pairs": int(len(pair_df)),
        "sklearn_auroc_final": float(sklearn_auroc_final),
        "pairwise_auroc_final": float(pairwise_auroc_final),
        "auroc_abs_diff": float(abs(sklearn_auroc_final - pairwise_auroc_final)),
        "flip_rate_mean": float(pair_df["flip"].mean()),
        "pair_margin_var_mean": float(pair_df["pair_margin_var"].mean()),
    }

    # Sample-level instability means by class
    anomaly_samples = sample_df[sample_df["image_label"] == "anomaly"]
    normal_samples = sample_df[sample_df["image_label"] == "normal"]
    if len(anomaly_samples) == 0 or len(normal_samples) == 0:
        raise ValueError("sample_df must contain both anomaly and normal samples.")

    summary.update(
        {
            "anomaly_I_bin_mean": float(anomaly_samples["I_bin"].mean()),
            "normal_I_bin_mean": float(normal_samples["I_bin"].mean()),
            "anomaly_I_cont_mean": float(anomaly_samples["I_cont"].mean()),
            "normal_I_cont_mean": float(normal_samples["I_cont"].mean()),
        }
    )

    return summary


def save_outputs(
    pair_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Save:
      - pairwise_table.csv
      - sample_instability_table.csv
      - summary.json
    """
    if not output_dir or not isinstance(output_dir, str):
        raise ValueError("output_dir must be a non-empty string")
    os.makedirs(output_dir, exist_ok=True)

    pair_path = os.path.join(output_dir, "pairwise_table.csv")
    sample_path = os.path.join(output_dir, "sample_instability_table.csv")
    summary_path = os.path.join(output_dir, "summary.json")

    pair_df.to_csv(pair_path, index=False)
    sample_df.to_csv(sample_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def run_full_analysis(csv_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Full pipeline:
      1) load original sample table
      2) build pair-level table
      3) build sample-level instability table
      4) compute summary
      5) save outputs
    """
    df = load_score_table(csv_path)
    pair_df = build_pairwise_table(df)
    sample_df = build_sample_instability_table(pair_df)
    summary = summarize_instability_results(df, pair_df, sample_df)
    save_outputs(pair_df, sample_df, summary, output_dir)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pairwise instability analysis from sample-level score table")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input sample-level CSV table")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write outputs")
    args = parser.parse_args()

    summary = run_full_analysis(args.csv_path, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

