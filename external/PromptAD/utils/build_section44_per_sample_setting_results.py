#!/usr/bin/env python3
"""
Build Section 4.4 sample-level table:
PromptAD/result_analysis/per_sample_setting_results.csv

Columns:
  dataset, class, k, seed, image_path, label, score, instability
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional

import pandas as pd


def _label_to_int(v) -> Optional[int]:
    if pd.isna(v):
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"normal", "good", "0"}:
            return 0
        if s in {"anomaly", "abnormal", "bad", "1"}:
            return 1
    try:
        x = int(float(v))
        if x in (0, 1):
            return x
    except Exception:
        pass
    return None


def _find_score_csv(promptad_root: str, row: pd.Series) -> str:
    dataset = str(row["dataset"])
    cls = str(row["class"])
    k = int(row["k"])
    seed = int(row["seed"])
    data_source = str(row.get("data_source", ""))
    run_name = str(row.get("run_name", ""))
    summary_path = str(row.get("summary_path", ""))

    if data_source == "result_round1" and run_name:
        p = os.path.join(promptad_root, "result_round1", dataset, f"k_{k}", "csv", f"{run_name}.csv")
        if os.path.isfile(p):
            return p

    # seed_search standard path
    if summary_path:
        job_root = os.path.dirname(os.path.dirname(summary_path))
        p = os.path.join(
            job_root,
            dataset,
            f"k_{k}",
            "csv",
            f"CLS-{dataset}-{cls}-k{k}-seed{seed}-per_sample.csv",
        )
        if os.path.isfile(p):
            return p

        # fallback glob inside job root
        patt = os.path.join(job_root, "**", f"CLS-{dataset}-{cls}-k{k}-seed{seed}-per_sample.csv")
        hit = sorted(glob.glob(patt, recursive=True))
        if hit:
            return hit[0]
    raise FileNotFoundError(f"score csv missing for {dataset}/{cls}/k{k}/seed{seed}")


def _find_instability_csv(row: pd.Series) -> str:
    summary_path = str(row.get("summary_path", ""))
    if not summary_path:
        raise FileNotFoundError("empty summary_path")
    p = os.path.join(os.path.dirname(summary_path), "sample_instability_table.csv")
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"sample_instability_table missing near {summary_path}")


def _load_score_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_path" not in df.columns:
        raise ValueError(f"{path}: missing image_path")

    label_col = "label" if "label" in df.columns else ("image_label" if "image_label" in df.columns else None)
    if label_col is None:
        raise ValueError(f"{path}: missing label/image_label")

    score_col = None
    for c in ("harmonic_score", "final_score", "score", "anomaly_score", "s_fused"):
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        raise ValueError(f"{path}: missing score-like column")

    out = df[["image_path", label_col, score_col]].copy()
    out = out.rename(columns={label_col: "label", score_col: "score"})
    out["label"] = out["label"].map(_label_to_int)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["image_path", "label", "score"]).copy()
    out["label"] = out["label"].astype(int)
    return out


def _load_instability_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_path" not in df.columns:
        raise ValueError(f"{path}: missing image_path")
    inst_col = None
    for c in ("instability", "I_bin", "flip_rate", "mean_I", "mean_instability"):
        if c in df.columns:
            inst_col = c
            break
    if inst_col is None:
        raise ValueError(f"{path}: missing instability-like column")
    out = df[["image_path", inst_col]].copy().rename(columns={inst_col: "instability"})
    out["instability"] = pd.to_numeric(out["instability"], errors="coerce")
    out = out.dropna(subset=["image_path", "instability"]).copy()
    # If instability table is pair-level by mistake, aggregate to sample-level.
    out = out.groupby("image_path", as_index=False)["instability"].mean()
    return out


def build_table(input_csv: str, output_csv: str, setting_count_csv: str) -> None:
    base = pd.read_csv(input_csv)
    needed = {"dataset", "class", "k", "seed", "summary_path"}
    missing = sorted(needed - set(base.columns))
    if missing:
        raise ValueError(f"input csv missing columns: {missing}")

    promptad_root = os.path.abspath(os.path.join(os.path.dirname(input_csv), ".."))
    rows: List[pd.DataFrame] = []
    for _, r in base.iterrows():
        score_csv = _find_score_csv(promptad_root, r)
        inst_csv = _find_instability_csv(r)
        score_df = _load_score_df(score_csv)
        inst_df = _load_instability_df(inst_csv)
        merged = score_df.merge(inst_df, on="image_path", how="inner", validate="one_to_one")
        merged["dataset"] = str(r["dataset"])
        merged["class"] = str(r["class"])
        merged["k"] = int(r["k"])
        merged["seed"] = int(r["seed"])
        rows.append(merged[["dataset", "class", "k", "seed", "image_path", "label", "score", "instability"]])

    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["dataset", "class", "k", "seed", "image_path"]).copy()
    all_df = all_df.sort_values(["dataset", "class", "k", "seed", "image_path"], kind="mergesort").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    all_df.to_csv(output_csv, index=False)

    setting_counts = (
        all_df.groupby(["dataset", "class", "k", "seed"], as_index=False)
        .size()
        .rename(columns={"size": "num_samples"})
        .sort_values(["dataset", "class", "k", "seed"], kind="mergesort")
        .reset_index(drop=True)
    )
    setting_counts.to_csv(setting_count_csv, index=False)

    print(f"total_samples={len(all_df)}")
    print(f"num_settings={setting_counts.shape[0]}")
    print(f"output={output_csv}")
    print(f"setting_counts={setting_count_csv}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        type=str,
        default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/per_seed_setting_metrics.csv",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/per_sample_setting_results.csv",
    )
    p.add_argument(
        "--setting-count-csv",
        type=str,
        default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/per_sample_setting_counts.csv",
    )
    args = p.parse_args()
    build_table(args.input_csv, args.output_csv, args.setting_count_csv)


if __name__ == "__main__":
    main()
