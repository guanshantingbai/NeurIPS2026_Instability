#!/usr/bin/env python3
"""
Section 4.4: decision consequence analysis (risk-coverage evaluation).

Pipeline:
1) Auto-discover candidate CSVs under PromptAD/result_analysis
2) Validate and map required fields:
   - setting_id (or dataset+class+k+seed)
   - score
   - instability
   - label
3) Compute pairwise ranking error per setting:
   error = 1[score(x+) <= score(x-)]
4) Evaluate rejection strategies at coverages in {0.9, 0.8, 0.7}:
   - random rejection (5 repeats)
   - score-based rejection (|score - median(score)| desc)
   - instability-based rejection (instability asc)
5) Aggregate mean risk across settings
6) Save:
   - decision_risk_per_setting.csv
   - decision_risk_summary.csv
7) Optional: risk-coverage curve figure
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SETTING_CANDIDATES: Sequence[str] = ("setting_id", "run_name", "setting", "config_id")
DATASET_CANDIDATES: Sequence[str] = ("dataset", "data", "dataset_name")
CLASS_CANDIDATES: Sequence[str] = ("class", "category", "cls", "object_class")
K_CANDIDATES: Sequence[str] = ("k", "shot", "fewshot_k")
SEED_CANDIDATES: Sequence[str] = ("seed", "random_seed")
SCORE_CANDIDATES: Sequence[str] = (
    "score",
    "final_score",
    "harmonic_score",
    "anomaly_score",
    "pred_score",
)
INSTABILITY_CANDIDATES: Sequence[str] = (
    "instability",
    "flip_rate",
    "mean_instability",
    "instability_score",
    "true_instability",
    "I",
    "I_x",
)
LABEL_CANDIDATES: Sequence[str] = ("label", "y", "target", "gt", "is_anomaly")


def round4(x: float) -> float:
    return float(f"{x:.4f}")


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        mapped = lower.get(name.lower())
        if mapped is not None:
            return mapped
    return None


def _resolve_setting_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    sid = _first_existing_column(df, SETTING_CANDIDATES)
    ds = _first_existing_column(df, DATASET_CANDIDATES)
    clz = _first_existing_column(df, CLASS_CANDIDATES)
    k = _first_existing_column(df, K_CANDIDATES)
    seed = _first_existing_column(df, SEED_CANDIDATES)
    return sid, ds, clz, k, seed


def _discover_csvs(result_analysis_dir: str) -> List[str]:
    preferred_patterns = [
        "*per_sample*.csv",
        "*merged*results*.csv",
        "*sample*level*results*.csv",
    ]
    files: List[str] = []
    for pat in preferred_patterns:
        files.extend(glob.glob(os.path.join(result_analysis_dir, "**", pat), recursive=True))
    if not files:
        files = glob.glob(os.path.join(result_analysis_dir, "**", "*.csv"), recursive=True)
    files = [os.path.normpath(p) for p in files if os.path.isfile(p)]
    return sorted(set(files))


def _to_binary_label(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        return x.apply(
            lambda v: np.nan
            if pd.isna(v)
            else (1 if float(v) == 1.0 else (0 if float(v) == 0.0 else np.nan))
        )
    lower = s.astype(str).str.strip().str.lower()
    pos = {"1", "anomaly", "abnormal", "positive", "pos", "true", "yes"}
    neg = {"0", "normal", "negative", "neg", "false", "no"}
    return lower.apply(lambda v: 1 if v in pos else (0 if v in neg else np.nan))


def _ranking_error(scores: np.ndarray, labels01: np.ndarray) -> float:
    pos_scores = scores[labels01 == 1]
    neg_scores = scores[labels01 == 0]
    n_pos = int(pos_scores.size)
    n_neg = int(neg_scores.size)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # AUROC by rank statistic with tie correction -> error = 1 - AUROC.
    s = pd.Series(np.concatenate([pos_scores, neg_scores]))
    y = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)])
    ranks = s.rank(method="average").to_numpy(dtype=np.float64)
    sum_rank_pos = float(np.sum(ranks[y == 1]))
    auc = (sum_rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    err = 1.0 - auc
    return float(err)


def _construct_setting_id(df: pd.DataFrame, sid_col: Optional[str], ds_col: Optional[str], clz_col: Optional[str], k_col: Optional[str], seed_col: Optional[str]) -> pd.Series:
    if sid_col is not None:
        return df[sid_col].astype(str)
    if ds_col and clz_col and k_col and seed_col:
        return (
            "CLS-"
            + df[ds_col].astype(str)
            + "-"
            + df[clz_col].astype(str)
            + "-k"
            + df[k_col].astype(str)
            + "-seed"
            + df[seed_col].astype(str)
        )
    raise ValueError("Missing setting identifier columns.")


def _pick_best_input(csv_paths: Sequence[str]) -> Tuple[str, Dict[str, Any], pd.DataFrame]:
    best: Optional[Tuple[str, Dict[str, Any], pd.DataFrame, int]] = None
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if len(df) == 0:
            continue
        sid, ds, clz, k, seed = _resolve_setting_columns(df)
        score = _first_existing_column(df, SCORE_CANDIDATES)
        instab = _first_existing_column(df, INSTABILITY_CANDIDATES)
        label = _first_existing_column(df, LABEL_CANDIDATES)
        setting_ok = sid is not None or all(x is not None for x in (ds, clz, k, seed))
        required_hits = int(setting_ok) + int(score is not None) + int(instab is not None) + int(label is not None)
        mapping = {
            "setting_id": sid,
            "score": score,
            "instability": instab,
            "label": label,
            "dataset": ds,
            "class": clz,
            "k": k,
            "seed": seed,
        }
        if best is None or required_hits > best[3] or (required_hits == best[3] and len(df) > len(best[2])):
            best = (p, mapping, df, required_hits)
        if required_hits == 4:
            # Early return on first fully valid file.
            return p, mapping, df
    if best is None:
        raise ValueError("No readable CSV found under result_analysis.")
    return best[0], best[1], best[2]


def _standardize_dataframe(df_raw: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
    sid_col = mapping.get("setting_id")
    ds_col = mapping.get("dataset")
    clz_col = mapping.get("class")
    k_col = mapping.get("k")
    seed_col = mapping.get("seed")
    score_col = mapping.get("score")
    instab_col = mapping.get("instability")
    label_col = mapping.get("label")
    if score_col is None or instab_col is None or label_col is None:
        raise ValueError("Missing required score/instability/label column.")
    if sid_col is None and not all(x is not None for x in (ds_col, clz_col, k_col, seed_col)):
        raise ValueError("Missing setting_id and missing fallback dataset+class+k+seed.")
    out = pd.DataFrame()
    out["setting_id"] = _construct_setting_id(df_raw, sid_col, ds_col, clz_col, k_col, seed_col)
    out["score"] = pd.to_numeric(df_raw[score_col], errors="coerce")
    out["instability"] = pd.to_numeric(df_raw[instab_col], errors="coerce")
    out["label"] = _to_binary_label(df_raw[label_col])
    out = out.dropna(subset=["setting_id", "score", "instability", "label"]).copy()
    out["label"] = out["label"].astype(int)
    out = out[(out["label"] == 0) | (out["label"] == 1)]
    return out


def _eval_one_setting(
    g: pd.DataFrame,
    coverages: Sequence[float],
    random_repeats: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    n = len(g)
    if n < 4:
        return []
    scores = g["score"].to_numpy(dtype=np.float64)
    instab = g["instability"].to_numpy(dtype=np.float64)
    labels = g["label"].to_numpy(dtype=np.int8)
    if np.sum(labels == 1) == 0 or np.sum(labels == 0) == 0:
        return []

    rows: List[Dict[str, Any]] = []
    sid = str(g["setting_id"].iloc[0])

    for cov in coverages:
        keep_n = max(2, int(math.floor(cov * n)))
        if keep_n >= n:
            keep_n = n

        rand_risks: List[float] = []
        for _ in range(random_repeats):
            idx = rng.choice(n, size=keep_n, replace=False)
            r = _ranking_error(scores[idx], labels[idx])
            if np.isfinite(r):
                rand_risks.append(float(r))
        random_risk = float(np.mean(rand_risks)) if rand_risks else float("nan")

        conf = np.abs(scores - np.median(scores))
        idx_score = np.argsort(-conf, kind="mergesort")[:keep_n]
        score_risk = _ranking_error(scores[idx_score], labels[idx_score])

        idx_instab = np.argsort(instab, kind="mergesort")[:keep_n]
        instab_risk = _ranking_error(scores[idx_instab], labels[idx_instab])

        rows.append(
            {
                "setting_id": sid,
                "n_samples": int(n),
                "coverage": round4(float(cov)),
                "risk_random": round4(random_risk) if np.isfinite(random_risk) else float("nan"),
                "risk_score": round4(score_risk) if np.isfinite(score_risk) else float("nan"),
                "risk_instability": round4(instab_risk) if np.isfinite(instab_risk) else float("nan"),
                "delta_instability_minus_score": (
                    round4(instab_risk - score_risk)
                    if np.isfinite(instab_risk) and np.isfinite(score_risk)
                    else float("nan")
                ),
            }
        )
    return rows


def _plot_risk_coverage(summary_df: pd.DataFrame, out_path: str) -> None:
    x = summary_df["coverage"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, summary_df["risk_random"].to_numpy(dtype=np.float64), marker="o", label="Random")
    ax.plot(x, summary_df["risk_score"].to_numpy(dtype=np.float64), marker="o", label="Score")
    ax.plot(x, summary_df["risk_instability"].to_numpy(dtype=np.float64), marker="o", label="Instability")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (pairwise ranking error)")
    ax.set_title("Risk-Coverage (mean across settings)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    result_analysis_dir: str,
    output_dir: str,
    coverages: Sequence[float],
    min_samples_per_setting: int,
    random_repeats: int,
    random_seed: int,
    do_plot: bool,
) -> int:
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = _discover_csvs(result_analysis_dir)
    if not csv_paths:
        print(f"[ERROR] No CSV found under: {result_analysis_dir}")
        return 1

    chosen_csv, mapping, df_raw = _pick_best_input(csv_paths)
    setting_ok = (mapping["setting_id"] is not None) or all(
        mapping.get(x) is not None for x in ("dataset", "class", "k", "seed")
    )
    score_ok = mapping["score"] is not None
    instab_ok = mapping["instability"] is not None
    label_ok = mapping["label"] is not None
    if not (setting_ok and score_ok and instab_ok and label_ok):
        print("[ERROR] Required fields missing. Stop.")
        print(f"[INFO] Selected CSV: {chosen_csv}")
        print("[INFO] Column mapping candidates:")
        setting_repr = (
            mapping.get("setting_id")
            if mapping.get("setting_id") is not None
            else "dataset+class+k+seed"
        )
        print(f"  setting_id -> {setting_repr}")
        print(f"  score      -> {mapping.get('score')}")
        print(f"  instability-> {mapping.get('instability')}")
        print(f"  label      -> {mapping.get('label')}")
        print(f"  total_rows -> {len(df_raw)}")
        # Best effort setting count if possible.
        try:
            sid = _construct_setting_id(
                df_raw,
                mapping.get("setting_id"),
                mapping.get("dataset"),
                mapping.get("class"),
                mapping.get("k"),
                mapping.get("seed"),
            )
            print(f"  n_settings -> {sid.nunique()}")
        except Exception:
            print("  n_settings -> unavailable")
        return 2

    df = _standardize_dataframe(df_raw, mapping)
    n_rows = len(df)
    n_settings = int(df["setting_id"].nunique())

    print(f"[INFO] Selected CSV: {chosen_csv}")
    print("[INFO] Column mapping:")
    setting_repr = (
        mapping.get("setting_id")
        if mapping.get("setting_id") is not None
        else "dataset+class+k+seed"
    )
    print(f"  setting_id -> {setting_repr}")
    print(f"  score      -> {mapping.get('score')}")
    print(f"  instability-> {mapping.get('instability')}")
    print(f"  label      -> {mapping.get('label')}")
    print(f"[INFO] total_rows={n_rows}, n_settings={n_settings}")

    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for sid, g in df.groupby("setting_id", sort=True):
        if len(g) < min_samples_per_setting:
            skipped.append(str(sid))
            continue
        out_rows = _eval_one_setting(
            g,
            coverages=coverages,
            random_repeats=random_repeats,
            rng=rng,
        )
        if out_rows:
            rows.extend(out_rows)
        else:
            skipped.append(str(sid))

    if not rows:
        print("[ERROR] No valid settings after filtering. Stop.")
        print(f"[INFO] skipped_settings={len(skipped)}")
        return 3

    per_setting = pd.DataFrame(rows)
    per_setting = per_setting.sort_values(["coverage", "setting_id"], ascending=[False, True]).reset_index(drop=True)
    per_setting.to_csv(os.path.join(output_dir, "decision_risk_per_setting.csv"), index=False)

    summary = (
        per_setting.groupby("coverage", as_index=False)[["risk_random", "risk_score", "risk_instability", "delta_instability_minus_score"]]
        .mean()
        .sort_values("coverage", ascending=False)
        .reset_index(drop=True)
    )
    for c in ("risk_random", "risk_score", "risk_instability", "delta_instability_minus_score"):
        summary[c] = summary[c].map(round4)
    summary.to_csv(os.path.join(output_dir, "decision_risk_summary.csv"), index=False)

    print("\nCoverage | Random | Score | Instability")
    for _, r in summary.iterrows():
        print(
            f"{r['coverage']:.1f}      | {r['risk_random']:.4f} | {r['risk_score']:.4f} | {r['risk_instability']:.4f}"
        )

    print("\n[INFO] Mean (instability - score) by coverage:")
    for _, r in summary.iterrows():
        print(f"  coverage={r['coverage']:.1f}: {r['delta_instability_minus_score']:.4f}")

    if do_plot:
        _plot_risk_coverage(summary, os.path.join(output_dir, "fig_risk_coverage.png"))
        logger.info("Saved plot: fig_risk_coverage.png")

    if skipped:
        with open(os.path.join(output_dir, "skipped_settings.txt"), "w", encoding="utf-8") as f:
            for sid in skipped:
                f.write(f"{sid}\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 4.4 decision consequence (risk-coverage) analysis.")
    parser.add_argument(
        "--result-analysis-dir",
        type=str,
        default="result_analysis",
        help="Directory containing candidate CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result_analysis/section_44_decision_consequence",
        help="Output directory for risk tables/plot",
    )
    parser.add_argument(
        "--coverages",
        type=float,
        nargs="+",
        default=[0.9, 0.8, 0.7],
        help="Coverage values, e.g. 0.9 0.8 0.7",
    )
    parser.add_argument("--min-samples-per-setting", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    base = os.path.join(os.path.dirname(__file__), "..")
    result_analysis_dir = args.result_analysis_dir
    if not os.path.isabs(result_analysis_dir):
        result_analysis_dir = os.path.normpath(os.path.join(base, result_analysis_dir))
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(base, output_dir))

    code = run(
        result_analysis_dir=result_analysis_dir,
        output_dir=output_dir,
        coverages=args.coverages,
        min_samples_per_setting=args.min_samples_per_setting,
        random_repeats=args.random_repeats,
        random_seed=args.random_seed,
        do_plot=not args.no_plot,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
