"""
Run planned analyses on pairwise_instability outputs.

Exp 1–2: pairwise_table.csv (tertile buckets, flip rate per bucket).
Exp 3: sample_instability_table.csv merged with original per_sample CSV (I_bin tertiles, harmonic by label).
Exp 4: aggregate all summary.json under a k_shot directory into one category-level table.
Exp 5: instability-aware rejection from ranking-based error (pairwise z_final), no threshold classifier.

See plan: Four experiments data mapping (exp 1–4); exp 5 = rejection / ranking error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from analyze_instability_behavior import run_exp6_all


def _tertile_labels(s: pd.Series) -> pd.Series:
    """
    Assign each row to low / mid / high tertile by rank (handles ties via rank first).
    """
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        raise ValueError("Cannot bucket: all values are NaN.")
    ranks = s.rank(method="first")
    # qcut on ranks gives roughly equal counts per bucket
    try:
        return pd.qcut(ranks, q=3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        # Too few unique values
        cats = pd.qcut(ranks, q=min(3, len(ranks.unique())), duplicates="drop")
        return cats


_BUCKET_ORDER = {"low": 0, "mid": 1, "high": 2}


def _attach_bucket_id_and_range(
    out: pd.DataFrame,
    d: pd.DataFrame,
    bucket_col: str,
    range_value_col: str,
) -> pd.DataFrame:
    """
    For each bucket row, add bucket_id (0,1,2 for low/mid/high) and bucket_range:
    string \"[min, max]\" of range_value_col within that bucket (empirical span).
    """
    bucket_ids: List[int] = []
    bucket_ranges: List[str] = []
    for _, r in out.iterrows():
        b = r[bucket_col]
        mask = d[bucket_col].astype(str) == str(b)
        sub = d.loc[mask, range_value_col]
        vmin = float(sub.min())
        vmax = float(sub.max())
        bucket_ranges.append(f"[{vmin:.8g}, {vmax:.8g}]")
        bucket_ids.append(_BUCKET_ORDER.get(str(b), -1))
    out = out.copy()
    out.insert(1, "bucket_id", bucket_ids)
    out.insert(2, "bucket_range", bucket_ranges)
    return out


def run_exp1_pair_conflict_flip(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Experiment 1: bucket by pair_conflict (tertiles), report flip rate per bucket.

    flip_rate = mean(flip) within bucket (unchanged definition).
    pair_count = number of pairs in bucket.
    conflict_min/max = empirical min/max of pair_conflict in bucket.
    m_final_mean/median = summary of m_final within bucket (auxiliary context).
    """
    if "pair_conflict" not in pair_df.columns or "flip" not in pair_df.columns:
        raise ValueError("pair_df must contain pair_conflict and flip")
    if "m_final" not in pair_df.columns:
        raise ValueError("pair_df must contain m_final for exp1 auxiliary stats")
    d = pair_df.copy()
    d["bucket"] = _tertile_labels(d["pair_conflict"])
    out = (
        d.groupby("bucket", observed=True)
        .agg(
            pair_count=("flip", "size"),
            flip_rate=("flip", "mean"),
            conflict_min=("pair_conflict", "min"),
            conflict_max=("pair_conflict", "max"),
            pair_conflict_mean=("pair_conflict", "mean"),
            m_final_mean=("m_final", "mean"),
            m_final_median=("m_final", "median"),
        )
        .reset_index()
    )
    out = _attach_bucket_id_and_range(out, d, "bucket", "pair_conflict")
    # Readable column order: bucket keys, counts, primary rate, conflict span, m_final stats
    col_order = [
        "bucket",
        "bucket_id",
        "bucket_range",
        "pair_count",
        "flip_rate",
        "conflict_min",
        "conflict_max",
        "pair_conflict_mean",
        "m_final_mean",
        "m_final_median",
    ]
    return out[[c for c in col_order if c in out.columns]]


def run_exp2_abs_m_final_flip(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Experiment 2: bucket by |m_final| (tertiles), report flip rate per bucket.
    Expectation: smaller |m_final| -> higher flip (boundary fragility).

    pair_count, flip_rate as in exp1 style; abs_m_final min/max in bucket;
    pair_conflict_mean/median as auxiliary context per bucket.
    """
    if "m_final" not in pair_df.columns or "flip" not in pair_df.columns:
        raise ValueError("pair_df must contain m_final and flip")
    if "pair_conflict" not in pair_df.columns:
        raise ValueError("pair_df must contain pair_conflict for exp2 auxiliary stats")
    d = pair_df.copy()
    d["abs_m_final"] = np.abs(d["m_final"].astype(float))
    d["bucket"] = _tertile_labels(d["abs_m_final"])
    out = (
        d.groupby("bucket", observed=True)
        .agg(
            pair_count=("flip", "size"),
            flip_rate=("flip", "mean"),
            abs_m_final_min=("abs_m_final", "min"),
            abs_m_final_max=("abs_m_final", "max"),
            abs_m_final_mean=("abs_m_final", "mean"),
            pair_conflict_mean=("pair_conflict", "mean"),
            pair_conflict_median=("pair_conflict", "median"),
        )
        .reset_index()
    )
    out = _attach_bucket_id_and_range(out, d, "bucket", "abs_m_final")
    col_order = [
        "bucket",
        "bucket_id",
        "bucket_range",
        "pair_count",
        "flip_rate",
        "abs_m_final_min",
        "abs_m_final_max",
        "abs_m_final_mean",
        "pair_conflict_mean",
        "pair_conflict_median",
    ]
    return out[[c for c in col_order if c in out.columns]]


def infer_per_sample_csv_path(analysis_dir: str) -> str:
    """
    analysis_dir: .../{dataset}/k_X/pairwise_instability/CLS-...-per_sample/
    per_sample:   .../{dataset}/k_X/csv/CLS-...-per_sample.csv
    """
    base = os.path.basename(os.path.normpath(analysis_dir))
    if not base.endswith("per_sample") and "per_sample" not in base:
        raise ValueError(
            f"analysis_dir basename should match per_sample run folder; got: {base}"
        )
    k_dir = os.path.dirname(os.path.dirname(analysis_dir))  # .../k_X
    csv_dir = os.path.join(k_dir, "csv")
    cand = os.path.join(csv_dir, f"{base}.csv")
    if not os.path.isfile(cand):
        raise FileNotFoundError(
            f"Inferred per_sample CSV not found: {cand}. Pass --per_sample_csv explicitly."
        )
    return cand


def run_exp3_ibin_harmonic_buckets(
    sample_instability_path: str,
    per_sample_csv_path: str,
    error_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Experiment 3 (risk): merge sample instability with per-sample scores; tertile I_bin;
    per bucket report sample_count, I_bin span, and harmonic means split by anomaly vs normal.

    harmonic_score is descriptive risk signal only unless --error_threshold is set;
    then optional error_rate_normal_harmonic_gt_threshold is appended (not a default metric).
    """
    samp = pd.read_csv(sample_instability_path)
    per = pd.read_csv(per_sample_csv_path)
    need_s = {"image_path", "image_label", "I_bin"}
    need_p = {"image_path", "harmonic_score", "image_label"}
    if not need_s.issubset(samp.columns):
        raise ValueError(f"sample table missing columns: {need_s - set(samp.columns)}")
    if not need_p.issubset(per.columns):
        raise ValueError(f"per_sample table missing columns: {need_p - set(per.columns)}")

    merged = samp.merge(
        per[["image_path", "harmonic_score"]],
        on="image_path",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(samp):
        raise ValueError(
            f"Merge mismatch: sample rows {len(samp)} vs merged {len(merged)}. Check image_path alignment."
        )

    merged["I_bin_bucket"] = _tertile_labels(merged["I_bin"])

    if hasattr(merged["I_bin_bucket"].dtype, "categories"):
        bucket_order: List[Any] = list(merged["I_bin_bucket"].cat.categories)
    else:
        bucket_order = sorted(
            merged["I_bin_bucket"].unique(),
            key=lambda x: _BUCKET_ORDER.get(str(x), 99),
        )

    rows: List[Dict[str, Any]] = []
    for bucket in bucket_order:
        sub = merged[merged["I_bin_bucket"].astype(str) == str(bucket)]
        if len(sub) == 0:
            continue
        rec: Dict[str, Any] = {
            "I_bin_bucket": str(bucket),
            "bucket_id": int(_BUCKET_ORDER.get(str(bucket), -1)),
            # Empirical span of I_bin values falling in this tertile bucket
            "bucket_range": f"[{float(sub['I_bin'].min()):.8g}, {float(sub['I_bin'].max()):.8g}]",
            "sample_count": int(len(sub)),
            "count_anomaly": int((sub["image_label"] == "anomaly").sum()),
            "count_normal": int((sub["image_label"] == "normal").sum()),
            "mean_harmonic_anomaly": float(sub.loc[sub["image_label"] == "anomaly", "harmonic_score"].mean())
            if (sub["image_label"] == "anomaly").any()
            else float("nan"),
            "mean_harmonic_normal": float(sub.loc[sub["image_label"] == "normal", "harmonic_score"].mean())
            if (sub["image_label"] == "normal").any()
            else float("nan"),
            "I_bin_min": float(sub["I_bin"].min()),
            "I_bin_max": float(sub["I_bin"].max()),
        }
        if error_threshold is not None:
            normals = sub[sub["image_label"] == "normal"]
            if len(normals) > 0:
                err = (normals["harmonic_score"].astype(float) > error_threshold).astype(float).mean()
                rec["error_rate_normal_harmonic_gt_threshold"] = float(err)
            else:
                rec["error_rate_normal_harmonic_gt_threshold"] = float("nan")
            rec["error_threshold_used"] = float(error_threshold)
        rows.append(rec)

    out = pd.DataFrame(rows)
    base_cols = [
        "I_bin_bucket",
        "bucket_id",
        "bucket_range",
        "sample_count",
        "count_anomaly",
        "count_normal",
        "mean_harmonic_anomaly",
        "mean_harmonic_normal",
        "I_bin_min",
        "I_bin_max",
    ]
    extra = [c for c in out.columns if c not in base_cols]
    return out[base_cols + extra]


def build_sample_ranking_error_table(
    pair_df: pd.DataFrame,
    sample_instability_df: pd.DataFrame,
    per_sample_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Step 1 (exp5): sample-level ranking error from pairwise_table only.

    z_final(i,j) = 1[s_final(x_i^+) > s_final(x_j^-)] (already in pairwise_table).
    error_pair(i,j) = 1 - z_final(i,j).

    For anomaly x_i^+: error = mean_j error_pair(i,j) = 1 - mean_j z_final.
    For normal x_j^-: error = mean_i error_pair(i,j) = 1 - mean_i z_final.

    Merges I_bin from sample_instability_table and harmonic_score from per-sample CSV
    (harmonic is auxiliary only; error is not defined from harmonic_score).
    """
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
    """Metrics on remaining sample set after I_bin-based rejection."""
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
    """
    Exp 5: reject top fraction of samples by I_bin (highest first); report ranking error on remainder.

    Rejection fractions: 10%, 20%, 30% of all samples (ceil count), plus baseline (no rejection).
    """
    os.makedirs(out_dir, exist_ok=True)
    si = pd.read_csv(sample_instability_path)
    per = pd.read_csv(per_sample_csv_path)

    sample_rank_df = build_sample_ranking_error_table(pair_df, si, per)
    p_sample = os.path.join(out_dir, "exp5_sample_ranking_error.csv")
    sample_rank_df.to_csv(p_sample, index=False)

    rows: List[Dict[str, Any]] = []
    rows.append({"setting": "baseline", **_rejection_split_metrics(sample_rank_df)})

    n = len(sample_rank_df)
    # Sort by I_bin descending; reject highest-I_bin samples first (stable for ties)
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


def parse_class_from_run_folder(folder_name: str) -> str:
    """
    folder_name like CLS-visa-candle-k2-seed111-per_sample or CLS-mvtec-bottle-k1-seed111-per_sample
    """
    m = re.match(r"CLS-(?:mvtec|visa)-(.+)-k\d+-seed\d+-per_sample$", folder_name)
    if m:
        return m.group(1)
    return folder_name


def run_exp4_aggregate_summaries(k_shot_dir: str) -> pd.DataFrame:
    """
    Collect summary.json under k_shot_dir/pairwise_instability/*/summary.json

    flip_rate_mean, num_samples, num_pairs, sklearn_auroc_final come from summary.json.
    flip_rate_mean_source documents that provenance.
    flip_rate_std is recomputed from pairwise_table.csv (population std, ddof=0) when present.
    """
    root = os.path.normpath(k_shot_dir)
    pattern_dir = os.path.join(root, "pairwise_instability")
    if not os.path.isdir(pattern_dir):
        raise FileNotFoundError(f"No pairwise_instability dir under: {root}")

    rows: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(pattern_dir)):
        sub = os.path.join(pattern_dir, name)
        if not os.path.isdir(sub):
            continue
        sj = os.path.join(sub, "summary.json")
        if not os.path.isfile(sj):
            continue
        with open(sj, "r", encoding="utf-8") as f:
            data = json.load(f)
        category = parse_class_from_run_folder(name)

        row: Dict[str, Any] = dict(data)
        row["run_folder"] = name
        row["category"] = category
        row["flip_rate_mean_source"] = "summary_json"

        pair_csv = os.path.join(sub, "pairwise_table.csv")
        if os.path.isfile(pair_csv):
            ptd = pd.read_csv(pair_csv, usecols=["flip"])
            # Std of binary flip over all pairs (same weight as summary mean over pairs)
            row["flip_rate_std"] = float(ptd["flip"].std(ddof=0))
        else:
            row["flip_rate_std"] = float("nan")

        a = data.get("anomaly_I_bin_mean")
        n = data.get("normal_I_bin_mean")
        if a is not None and n is not None:
            row["mean_I_bin_macro"] = (float(a) + float(n)) / 2.0

        rows.append(row)

    if not rows:
        raise ValueError(f"No summary.json found under {pattern_dir}/*/")

    df = pd.DataFrame(rows)
    preferred = [
        "category",
        "run_folder",
        "num_samples",
        "num_pairs",
        "flip_rate_mean",
        "flip_rate_std",
        "flip_rate_mean_source",
        "sklearn_auroc_final",
        "pairwise_auroc_final",
        "auroc_abs_diff",
        "mean_I_bin_macro",
        "anomaly_I_bin_mean",
        "normal_I_bin_mean",
        "anomaly_I_cont_mean",
        "normal_I_cont_mean",
        "pair_margin_var_mean",
        "num_anomaly",
        "num_normal",
    ]
    front = [c for c in preferred if c in df.columns]
    rest = sorted([c for c in df.columns if c not in front])
    return df[front + rest]


_EXPECTED_EXPERIMENT_ARTIFACTS = (
    "exp1_pair_conflict_flip_rate.csv",
    "exp2_abs_m_final_flip_rate.csv",
    "exp3_ibin_harmonic_by_bucket.csv",
    "exp3_meta.json",
    "exp4_category_summary.csv",
    "exp5_sample_ranking_error.csv",
    "exp5_instability_rejection.csv",
    "exp6_extreme_samples.csv",
    "exp6_ibin_error_bucket.csv",
    "exp6_ibin_vs_error_scatter.png",
    "exp6_rejection_curve.png",
)


def _is_standard_cls_run_folder(folder_name: str) -> bool:
    """Skip stray dirs (e.g. legacy 'capsules') that are not CLS-*-per_sample runs."""
    return folder_name.startswith("CLS-") and "per_sample" in folder_name


def discover_pairwise_analysis_dirs(result_root: str) -> List[str]:
    """Each dir contains pairwise_table.csv (and expected: sample_instability_table.csv)."""
    pattern = os.path.join(
        result_root, "*", "k_*", "pairwise_instability", "*", "pairwise_table.csv"
    )
    paths = sorted(glob.glob(pattern))
    out: List[str] = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        run_dir = os.path.dirname(p)
        if _is_standard_cls_run_folder(os.path.basename(run_dir)):
            out.append(run_dir)
    return out


def k_shot_dir_from_analysis(analysis_dir: str) -> str:
    """.../{dataset}/k_X/pairwise_instability/RUN -> .../{dataset}/k_X"""
    return os.path.dirname(os.path.dirname(os.path.normpath(analysis_dir)))


def _experiments_dir_complete(exp_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(exp_dir, f)) for f in _EXPECTED_EXPERIMENT_ARTIFACTS)


def _broadcast_exp4_to_runs(k_shot_dir: str, exp4_src: str) -> None:
    """Copy k-shot exp4 aggregate into every run's experiments/ (matches per-run layout)."""
    pi = os.path.join(os.path.normpath(k_shot_dir), "pairwise_instability")
    if not os.path.isdir(pi):
        return
    for name in sorted(os.listdir(pi)):
        run_dir = os.path.join(pi, name)
        if not os.path.isdir(run_dir):
            continue
        exp_dir = os.path.join(run_dir, "experiments")
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy2(exp4_src, os.path.join(exp_dir, "exp4_category_summary.csv"))


def run_exp4_write_and_broadcast(k_shot_dir: str) -> Optional[str]:
    """
    Build exp4 from all summary.json under k_shot_dir/pairwise_instability/*,
    write k_shot_dir/experiments/exp4_category_summary.csv, copy into each run's experiments/.
    """
    k_shot_dir = os.path.normpath(k_shot_dir)
    df = run_exp4_aggregate_summaries(k_shot_dir)
    out_dir = os.path.join(k_shot_dir, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    p4 = os.path.join(out_dir, "exp4_category_summary.csv")
    df.to_csv(p4, index=False)
    _broadcast_exp4_to_runs(k_shot_dir, p4)
    return p4


def run_batch_fill_experiments(
    result_root: str,
    *,
    skip_existing: bool = False,
    quiet_exp6: bool = True,
    error_threshold: Optional[float] = None,
) -> Tuple[int, int, List[str]]:
    """
    For every pairwise_instability run under result_root: exp1–3, exp5, exp6; then per k_shot exp4 + copy.

    Returns (n_ok, n_fail, errors).
    """
    result_root = os.path.abspath(os.path.normpath(result_root))
    if not os.path.isdir(result_root):
        raise FileNotFoundError(result_root)

    analysis_dirs = discover_pairwise_analysis_dirs(result_root)
    errors: List[str] = []
    n_ok = 0
    n_fail = 0

    for analysis_dir in analysis_dirs:
        exp_dir = os.path.join(analysis_dir, "experiments")
        if skip_existing and os.path.isdir(exp_dir) and _experiments_dir_complete(exp_dir):
            print(f"[skip] {analysis_dir}")
            continue

        sample_inst = os.path.join(analysis_dir, "sample_instability_table.csv")
        if not os.path.isfile(sample_inst):
            msg = f"Missing sample_instability_table.csv: {analysis_dir}"
            warnings.warn(msg)
            errors.append(msg)
            n_fail += 1
            continue

        try:
            run_single_analysis_dir(
                analysis_dir,
                per_sample_csv=None,
                error_threshold=error_threshold,
            )
            run_exp6_all(exp_dir, quiet=quiet_exp6)
        except Exception as e:
            msg = f"{analysis_dir}: {e}"
            warnings.warn(msg)
            errors.append(msg)
            n_fail += 1
            continue

        print(f"[ok]   {analysis_dir}")
        n_ok += 1

    k_shots: Set[str] = {k_shot_dir_from_analysis(d) for d in analysis_dirs}
    for kdir in sorted(k_shots):
        try:
            p4 = run_exp4_write_and_broadcast(kdir)
            print(f"[exp4] {kdir} -> {p4}")
        except Exception as e:
            msg = f"exp4 failed for {kdir}: {e}"
            warnings.warn(msg)
            errors.append(msg)

    return n_ok, n_fail, errors


def run_single_analysis_dir(
    analysis_dir: str,
    per_sample_csv: Optional[str] = None,
    error_threshold: Optional[float] = None,
) -> Dict[str, str]:
    """
    Run exp 1–3 and exp 5 for one pairwise_instability output folder. Writes CSVs under analysis_dir/experiments/.
    Returns paths written.
    """
    analysis_dir = os.path.normpath(analysis_dir)
    pair_path = os.path.join(analysis_dir, "pairwise_table.csv")
    sample_path = os.path.join(analysis_dir, "sample_instability_table.csv")
    if not os.path.isfile(pair_path):
        raise FileNotFoundError(pair_path)
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(sample_path)

    out_dir = os.path.join(analysis_dir, "experiments")
    os.makedirs(out_dir, exist_ok=True)

    pair_df = pd.read_csv(pair_path)

    exp1 = run_exp1_pair_conflict_flip(pair_df)
    p1 = os.path.join(out_dir, "exp1_pair_conflict_flip_rate.csv")
    exp1.to_csv(p1, index=False)

    exp2 = run_exp2_abs_m_final_flip(pair_df)
    p2 = os.path.join(out_dir, "exp2_abs_m_final_flip_rate.csv")
    exp2.to_csv(p2, index=False)

    per_csv = per_sample_csv or infer_per_sample_csv_path(analysis_dir)
    exp3 = run_exp3_ibin_harmonic_buckets(sample_path, per_csv, error_threshold=error_threshold)
    p3 = os.path.join(out_dir, "exp3_ibin_harmonic_by_bucket.csv")
    exp3.to_csv(p3, index=False)

    meta = {
        "per_sample_csv_used": per_csv,
        "error_threshold": error_threshold,
    }
    with open(os.path.join(out_dir, "exp3_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    p5_sample, p5_rej = run_instability_rejection_experiment(
        pair_df, sample_path, per_csv, out_dir
    )

    return {
        "exp1": p1,
        "exp2": p2,
        "exp3": p3,
        "exp5_sample_ranking_error": p5_sample,
        "exp5_instability_rejection": p5_rej,
        "experiments_dir": out_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Instability experiments 1–5 on pairwise_instability outputs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="Run exp 1–3 and exp 5 for one analysis folder")
    p_single.add_argument(
        "--analysis_dir",
        type=str,
        required=True,
        help="Folder containing pairwise_table.csv and sample_instability_table.csv",
    )
    p_single.add_argument(
        "--per_sample_csv",
        type=str,
        default=None,
        help="Override path to original CLS-*-per_sample.csv (default: infer from ../../csv/)",
    )
    p_single.add_argument(
        "--error_threshold",
        type=float,
        default=None,
        help="If set, exp3 adds error_rate_normal_harmonic_gt_threshold per bucket",
    )

    p_agg = sub.add_parser("aggregate", help="Run exp 4: merge all summary.json under k_shot_dir")
    p_agg.add_argument(
        "--k_shot_dir",
        type=str,
        required=True,
        help="e.g. result_round1/visa/k_2",
    )

    p_batch = sub.add_parser(
        "batch",
        help="Fill experiments/ for all pairwise_instability runs (exp1–6 + exp4 broadcast)",
    )
    p_batch.add_argument(
        "--result_root",
        type=str,
        default=os.path.join("PromptAD", "result_round1"),
        help="Directory containing dataset/k_*/pairwise_instability/...",
    )
    p_batch.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs whose experiments/ already has all expected artifacts",
    )
    p_batch.add_argument(
        "--verbose_exp6",
        action="store_true",
        help="Print per-run exp6 task logs (default: quiet)",
    )
    p_batch.add_argument(
        "--error_threshold",
        type=float,
        default=None,
        help="Forwarded to exp3 (optional error-rate column)",
    )

    args = parser.parse_args()

    if args.cmd == "single":
        paths = run_single_analysis_dir(
            args.analysis_dir,
            per_sample_csv=args.per_sample_csv,
            error_threshold=args.error_threshold,
        )
        print(json.dumps(paths, indent=2))
    elif args.cmd == "aggregate":
        df = run_exp4_aggregate_summaries(args.k_shot_dir)
        out_dir = os.path.join(os.path.normpath(args.k_shot_dir), "experiments")
        os.makedirs(out_dir, exist_ok=True)
        p4 = os.path.join(out_dir, "exp4_category_summary.csv")
        df.to_csv(p4, index=False)
        print(f"Wrote {p4} ({len(df)} categories)")
    else:
        n_ok, n_fail, errs = run_batch_fill_experiments(
            args.result_root,
            skip_existing=args.skip_existing,
            quiet_exp6=not args.verbose_exp6,
            error_threshold=args.error_threshold,
        )
        print(json.dumps({"ok": n_ok, "fail": n_fail, "num_errors": len(errs)}, indent=2))


if __name__ == "__main__":
    main()
